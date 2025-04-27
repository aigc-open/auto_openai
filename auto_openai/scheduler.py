import concurrent.futures
import time
import random
from auto_openai.utils.public import scheduler
from auto_openai.utils.openai import Scheduler, RedisStreamInfer, VLLMOpenAI
from loguru import logger
from auto_openai.utils.init_env import global_config
from auto_openai.utils.support_models.model_config import system_models_config
from auto_openai.utils.backends import ComfyuiTask, WebuiTask, MaskGCTTask, FunAsrTask, \
    EmbeddingTask, LLMTramsformerTask, RerankTask, DiffusersVideoTask, HttpLLMTask, Wan21Task


class Task(ComfyuiTask, WebuiTask, MaskGCTTask, FunAsrTask,
           EmbeddingTask, LLMTramsformerTask, RerankTask, 
           DiffusersVideoTask, Wan21Task, HttpLLMTask):

    def loop_infer(self, llm_server, request_info, free_status_list, idx, max_workers=8, infer_fn=None):
        logger.info(f"进程服务信息: {llm_server}")
        if request_info:
            free_status_list[idx] = False  # 改线程非空闲
            request_id = request_info.get("request_id")
            params = request_info.get("params")

        times = 0
        unuseful_times = 0
        future_to_task = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            if request_info:
                future_to_task.add(executor.submit(
                    infer_fn, llm_server, request_id, params, self.model_config))
            while True:
                if sum(not task.done() for task in future_to_task) > 0:
                    # 任务已被处理完，更新当前线程状态
                    free_status_list[idx] = False
                else:
                    free_status_list[idx] = True
                self.update_running_model()

                if unuseful_times >= self.unuseful_times and all(free_status_list) or times > self.useful_times:
                    # 当有效次数达到预期
                    # 当所有线程空闲时且无用次数达到预期
                    break
                request_info = self.get_request(model_name=self.current_model)
                if not request_info:
                    # 该模型没有推理任务，则跳过该模型
                    # time.sleep(1)
                    unuseful_times += 1
                    free_status_list[idx] = True
                    continue
                times += 1
                unuseful_times = 0  # 当获取到新的任务时，将无效次数重置
                free_status_list[idx] = False  # 改线程非空闲
                request_id = request_info.get("request_id")
                params = request_info.get("params")
                while sum(not task.done() for task in future_to_task) > max_workers:
                    # 如果任务太多了，停止加任务,等其处理完
                    time.sleep(2)
                    pass
                future_to_task.add(executor.submit(
                    infer_fn, llm_server, request_id, params, self.model_config))
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]

    def run(self):
        self.update_running_model()
        for model_name in self.loop_model():  # 获取有任务的队列的模型出来
            self.model_config = scheduler.get_model_config(
                model_name=model_name)  # 设置本次需要运行的模型参数
            if not self.model_config:
                # 如果没有该模型配置，则跳过该模型
                continue
            # 检测模型文件是否存在,如果不存在就直接跳过该模型
            try:
                if not system_models_config.get(model_name).is_available():
                    # 如果没有该模型配置，则跳过该模型, 说明该模型不在该调度器中运行
                    # 服务类型不在范围内得也要跳过
                    continue
            except:
                continue
            self.set_infer_fn()
            # 更新当前设备的gpu count
            if self.model_config.get("gpu_types", {}).get(global_config.GPU_TYPE):
                self.model_config["need_gpu_count"] = self.model_config.get(
                    "gpu_types", {}).get(global_config.GPU_TYPE).get("need_gpu_count")
            else:
                # 不希望该模型运行在此设备上,让别的设备运行此模型
                continue
            if self.model_config["need_gpu_count"] > len(self.node_gpu_total):
                # 对于需要卡资源大于节点卡数的,这个队列不处理,表示该调度器无法处理该模型,放给其他调度器处理
                if int(time.time()) % 100 == 0:
                    pass
                    # logger.info(f'模型: {self.model_config["name"]} 所需资源太多，放给其他调度器处理')
                continue

            self.update_running_model()
            if self.current_model is not None and self.current_model != self.model_config["name"]:

                if model_name in scheduler.get_running_model() and scheduler.get_request_queue_length(model_name=model_name) <= self.max_workers*2:
                    # 已存在的任务完全能自己处理了，不用启动新任务了
                    continue
                # 当模型和自己模型不一致时，尽可能其正在运行的模型取走，如果取不走，说明并发大,需要新增副本
                time.sleep(random.randint(1, 5))
                if self.model_config["need_gpu_count"] != len(self.node_gpu_total):
                    # 尽量让其他对等节点去推理
                    time.sleep(random.randint(1, 5))
            elif self.current_model is None:
                # 如果当前模型为空，但是该模型正在运行，则等待
                if model_name in scheduler.get_running_model() and scheduler.get_request_queue_length(model_name=model_name) <= self.max_workers*2:
                    # 已存在的任务完全能自己处理了，不用启动新任务了
                    continue

                # time.sleep(random.randint(1, 5))

            request_info = self.get_request(model_name=model_name)
            if not request_info:
                logger.info(f'模型: {self.model_config["name"]} 没有推理请求, 跳过')
                # 该模型没有推理任务，则跳过该模型
                continue

            #################### 启动推理服务 ####################
            if self.current_model != self.model_config["name"]:
                self.start_server()

            first_request_info = [None]*self.workers_num
            first_request_info[0] = request_info
            free_status_list = [True]*self.workers_num  # 线程是否空闲

            if self.workers_num == 0:
                request_id = request_info.get("request_id")
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(text="资源不足，无法启动该模型[500]", finish=True, usage=dict(completion_tokens=0,
                                                                                                              prompt_tokens=0, total_tokens=0, tps=0)))
                continue

            #################### 启动推理任务 ####################
            self.set_infer_fn()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                # 提交任务给线程池,线程池大小由实际任务数量决定
                future_to_task = {executor.submit(
                    self.loop_infer, self.service_list[i], first_request_info[i], free_status_list, i, self.max_workers, self.infer_fn): i for i in range(self.workers_num)}
                results = [
                    future.result() for future in concurrent.futures.as_completed(future_to_task)]

    def start_server(self):
        start_time = time.time()
        if "vllm" in self.model_config.get("server_type"):
            # 启动vllm 大模型服务
            self.start_vllm_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "comfyui":
            # 启动comfyui 大模型服务
            self.start_comfyui_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "webui":
            # 启动comfyui 大模型服务
            self.start_webui_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "maskgct":
            # 启动comfyui 大模型服务
            self.start_maskgct_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "funasr":
            # 启动funasr 大模型服务
            self.start_funasr_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "embedding":
            # 启动embedding 大模型服务
            self.start_embedding_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "llm-transformer-server":
            # 启动embedding 大模型服务
            self.start_llm_transformer_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "rerank":
            # 启动embedding 大模型服务
            self.start_rerank_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "diffusers-video":
            # 启动embedding 大模型服务
            self.start_diffusers_video_server(
                model_name=self.model_config["name"])
            
        elif self.model_config.get("server_type") == "wan21":
            # 启动 embedding 大模型服务
            self.start_wan21_server(
                model_name=self.model_config["name"])
        elif self.model_config.get("server_type") == "http-llm":
            # 启动embedding 大模型服务
            self.start_http_llm_server(
                model_name=self.model_config["name"])
        else:
            raise Exception(
                f"未知的模型服务类型: {self.model_config.get('server_type')}")
        end_time = time.time()
        # 收集模型启动时间
        self.profiler_collector(
            self.model_config["name"], "start_server_time", end_time - start_time, description="模型启动时间")

    def set_infer_fn(self):
        if "vllm" in self.model_config.get("server_type"):
            # 启动vllm 大模型服务
            self.max_workers = 8
            self.infer_fn = self.vllm_infer
            self.service_list = self.__service_list__(
                url_format="http://localhost:{port}/v1")
        elif self.model_config.get("server_type") == "comfyui":
            # 启动comfyui 大模型服务
            self.max_workers = 1
            self.infer_fn = self.comfyui_infer
            self.service_list = self.__service_list__(
                url_format="localhost:{port}")
        elif self.model_config.get("server_type") == "webui":
            # 启动comfyui 大模型服务
            self.max_workers = 1
            self.infer_fn = self.webui_infer
            self.service_list = self.__service_list__(
                url_format="localhost:{port}")
        elif self.model_config.get("server_type") == "maskgct":
            # 启动maskgct 大模型服务
            self.max_workers = 1
            self.infer_fn = self.maskgct_infer
            self.service_list = self.__service_list__(
                url_format="http://localhost:{port}")
        elif self.model_config.get("server_type") == "funasr":
            # 启动funasr 大模型服务
            self.max_workers = 1
            self.infer_fn = self.funasr_infer
            self.service_list = self.__service_list__(
                url_format="http://localhost:{port}")
        elif self.model_config.get("server_type") == "embedding":
            # 启动embedding 大模型服务
            self.max_workers = 1
            self.infer_fn = self.embedding_infer
            self.service_list = self.__service_list__(
                url_format="http://localhost:{port}")
        elif self.model_config.get("server_type") == "llm-transformer-server":
            # 启动 llm-transformer-server 大模型服务
            self.max_workers = 1
            self.infer_fn = self.llm_transformer_infer
            self.service_list = self.__service_list__(
                url_format="http://localhost:{port}/v1")

        elif self.model_config.get("server_type") == "rerank":
            self.max_workers = 1
            self.infer_fn = self.rerank_infer
            self.service_list = self.__service_list__(
                url_format="http://localhost:{port}")
        elif self.model_config.get("server_type") == "diffusers-video":
            self.max_workers = 1
            self.infer_fn = self.diffusers_video_infer
            self.service_list = self.__service_list__(
                url_format="http://localhost:{port}")
        elif self.model_config.get("server_type") == "wan21":
            self.max_workers = 1
            self.infer_fn = self.wan21_infer
            self.service_list = self.__service_list__(
                url_format="http://localhost:{port}")
        elif self.model_config.get("server_type") == "http-llm":
            # 启动embedding 大模型服务
            self.max_workers = 10
            self.infer_fn = self.http_llm_infer
            self.service_list = [self.model_config["base_url"] for i in range(self.workers_num)]
        else:
            raise Exception(
                f"未知的模型服务类型: {self.model_config.get('server_type')}")


if __name__ == "__main__":
    Task().loop()
