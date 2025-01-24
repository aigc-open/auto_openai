from auto_openai.utils.public import scheduler
from auto_openai.utils.openai import Scheduler, RedisStreamInfer, VLLMOpenAI
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import concurrent.futures
from loguru import logger
from auto_openai.utils.init_env import global_config
import json
from openai import OpenAI
import time
import subprocess
import requests
import os
import random
from auto_openai.utils.mock import LLM_MOCK_DATA
from auto_openai.utils.comfyui_client import ComfyUIClient
from auto_openai.utils.webui_client import WebUIClient
from auto_openai.utils.public import s3_client
import wget
from auto_openai.lm_server import CMD
from gradio_client import Client, file
from auto_openai.utils.cut_messages import messages_token_count, string_token_count, cut_string
from auto_openai.utils.daily_basic_function import safe_dir
import wget
import os
from auto_openai.utils.check_node import get_address_hostname
from pydub import AudioSegment
from auto_openai.utils.openai.openai_request import SD15MultiControlnetGenerateImage, ErrorResponse
import webuiapi
from auto_openai.utils.support_models.model_config import system_models_config
root_path = os.path.dirname(os.path.abspath(__file__))


MODEL_PROF_KEY = "Profiler"


class BaseTask:
    current_model = None
    worker_start_port = global_config.LM_SERVER_BASE_PORT
    model_config = {}
    useful_times = global_config.USERFULL_TIMES_PER_MODEL
    node_gpu_total = global_config.get_gpu_list()
    unuseful_times = global_config.UNUSERFULL_TIMES_PER_MODEL
    status = True

    def split_gpu(self):
        # lst = list(range(self.node_gpu_total))
        lst = self.node_gpu_total
        num_parts = self.workers_num
        if num_parts == 1:
            return [lst]
        else:
            split_size = len(lst) // num_parts
            return [lst[i:i+split_size] for i in range(0, len(lst), split_size)]

    def get_request(self, model_name):
        request_id = scheduler.get_request_queue(model_name)
        if not request_id:
            # logger.warning(f"{model_name} 的未取到request_id")
            return None
        start_time = time.time()
        task_params = None
        while time.time() - start_time > 100:
            # 3秒未获取到参数则视为被放弃的请求
            task_params = scheduler.get_request_params(request_id=request_id)
            if task_params:
                break
        if not task_params:
            logger.warning(f"{model_name} 的 {request_id} 参数为空")
            return None
        task_params.update({"stream": True})
        scheduler.set_request_status_ing(request_id=request_id)
        return {"request_id": request_id, "params": task_params}

    @property
    def workers_num(self):
        return len(self.node_gpu_total)//self.model_config["need_gpu_count"]

    def loop_model(self):
        """每次循环尽可能复用本地已经存在的模型"""
        model_list = scheduler.get_request_queue_names()
        if self.current_model and self.current_model in model_list:
            logger.info(
                f"本次循环第一顺位模型: {self.current_model} 服务状态: {self.status}")
            model_list.remove(self.current_model)
            model_list.insert(0, self.current_model)
        return model_list

    def update_running_model(self):
        if self.current_model:
            scheduler.set_running_model(model_name=self.current_model)

    def report_node(self):
        while True:
            try:
                data = get_address_hostname()
                data["device-ids"] = self.node_gpu_total
                data["device-type"] = global_config.GPU_TYPE
                scheduler.set_running_node(node_name=data.get(
                    "hostname"), value=json.dumps(data))
                time.sleep(8)
            except:
                pass

    def report_models(self):
        while True:
            try:
                for m in system_models_config.list():
                    if m.is_available() is True:
                        scheduler.set_available_model(
                            model_name=m.name, value=json.dumps(m.dict()))
                time.sleep(2)
            except Exception as e:
                pass

    def loop(self):
        self.kill_model_server()
        import threading
        logger.info("上报节点...")
        threading.Thread(target=self.report_node).start()
        logger.info("上报模型...")
        threading.Thread(target=self.report_models).start()
        logger.info("开始调度...")
        while True:
            try:
                self.run()
            except Exception as e:
                logger.exception("出现调度异常: {e}")

    def kill_model_server(self):
        CMD.kill()
        self.current_model = None

    def __service_list__(self, url_format: str = "http://localhost:{port}/v1"):
        """获取本地vllm进程服务请求接口"""
        result = []
        url = url_format
        for i in range(self.workers_num):
            result.append(url.format(port=self.worker_start_port + i))
        return result

    def get_audio_time(self, audio_path):

        if audio_path.startswith("http"):
            with safe_dir("temp") as _dir:
                audio_path_ = os.path.join(_dir, "temp_audio.mp3")
                wget.download(audio_path, audio_path_)
                # 加载音频文件
                audio = AudioSegment.from_file(audio_path_)  # 替换为你的音频文件路径
        else:
            # 加载音频文件
            audio = AudioSegment.from_file(audio_path)

        # 获取音频长度（以毫秒为单位）
        duration_ms = len(audio)

        # 转换为秒
        duration_sec = duration_ms / 1000.0
        return duration_sec

    def profiler_collector(self, model_name, key, value, description=""):
        if key not in "start_server_time":
            scheduler.set_request_done()
        device_model_name = f"{global_config.GPU_TYPE}/{model_name}"
        # 统计服务加载时间
        profiler = scheduler.get_profiler()
        server_time = profiler.get(key, {})
        # 添加最小时间，最大时间
        max_time = server_time.get(device_model_name, {}).get(
            "max_time", value)
        min_time = server_time.get(device_model_name, {}).get(
            "min_time", value)

        server_time.update({
            device_model_name: {
                "max_time": max(max_time, value),
                "min_time": min(min_time, value),
                "new_time": value,
                "description": description
            }
        })
        profiler[key] = server_time
        scheduler.set_profiler(data=profiler)


class VllmTask(BaseTask):

    def get_chat_template(self, model_name: str):
        self.stop_params = {}
        self.stop_params = {"stop": self.model_config.get("stop", [])}
        return f"/template/{self.model_config.get('template')}"

    def read_template(self, model_name: str):
        with open(self.get_chat_template(model_name), "r") as f:
            return f.read()

    def start_vllm(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
            status = True
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            if "NV" in global_config.GPU_TYPE:
                device_name = "auto"
            else:
                device_name = "gcu"
            try:
                server_type = self.model_config.get("server_type", "vllm")
                status = CMD.get_vllm(model_name=model_name, device=device, need_gpu_count=len(
                    self.split_gpu()[idx]), port=port, template=self.get_chat_template(model_name),
                    model_max_tokens=self.model_config['model_max_tokens'], device_name=device_name,
                    quantization=self.model_config.get("quantization", None), server_type=server_type)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_vllm_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_vllm, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def vllm_infer(self, llm_server, request_id, params, model_config):
        error = self.llm_pipeline(llm_server, request_id, params, model_config)
        if error is not None and type(error) == ErrorResponse and error.diff_max_tokens > 0:
            new_max_tokens = params["max_tokens"] - error.diff_max_tokens
            if new_max_tokens > 0:
                params["max_tokens"] = new_max_tokens
                error = self.llm_pipeline(
                    llm_server, request_id, params, model_config)
        if type(error) == ErrorResponse:
            logger.error(f"大模型推理错误: {llm_server} {request_id} {params}")
            scheduler.set_result(request_id=request_id,
                                 value=RedisStreamInfer(
                                     text=f"{error.message}",
                                     finish=True,
                                     usage=dict(prompt_tokens=0, total_tokens=0,
                                                completion_tokens=0, tps=0)))

    def llm_pipeline(self, llm_server, request_id, params, model_config):
        try:
            logger.info(f"处理大模型推理任务中: {llm_server} {request_id} {params}")
            if global_config.MOCK:
                send_text = ""
                for text in LLM_MOCK_DATA:
                    send_text += text
                    self.update_running_model()
                    if len(send_text) > 5:
                        scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                            text=f"{send_text}", finish=False))
                        send_text = ""
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{request_id} {params['model']} 推理完成",
                                         finish=True,
                                         usage=dict(prompt_tokens=10, total_tokens=10,
                                                    completion_tokens=10, tps=22)))
                time.sleep(1)
            else:
                if self.status == False:
                    raise Exception("模型服务启动异常")
                self.update_running_model()
                client = VLLMOpenAI(api_key="xxx", base_url=llm_server)
                params.update(self.stop_params)
                start_time = time.time()
                if params.get("messages"):
                    stream = client.chat.completions.create(**params)
                else:
                    stream = client.completions.create(**params)

                # 基本参数
                text = ""
                finish_reason = None
                finish = False
                completion_tokens = 0
                prompt_tokens = 0
                total_tokens = 0
                tps = 0
                send_text = ""
                all_text = ""
                pushed = False
                tool_call_function_name = ""
                tool_call_function_args = ""
                for chunk in stream:
                    self.update_running_model()
                    if hasattr(chunk, "code") and type(chunk) == ErrorResponse and chunk.code >= 300:
                        return chunk
                    if not pushed:
                        logger.info(
                            f"模型{params['model']}数据生成中...: {request_id}")
                        pushed = True

                    if params.get("messages") and chunk.choices[0].delta.tool_calls:
                        tool_call = chunk.choices[0].delta.tool_calls[0]
                        if tool_call.function.name:
                            tool_call_function_name = tool_call.function.name
                        if tool_call.function.arguments:
                            tool_call_function_args += tool_call.function.arguments
                    elif params.get("messages") and chunk.choices[0].delta.content is not None:
                        text = chunk.choices[0].delta.content
                    elif params.get("prompt") and chunk.choices[0].text is not None:
                        text = chunk.choices[0].text
                    else:
                        text = ""

                    if chunk.choices[0].finish_reason:
                        finish = True
                        finish_reason = chunk.choices[0].finish_reason
                        logger.info(
                            f"本轮对话{request_id}-finish_reason: {finish_reason}")
                        end_time = time.time()
                        if chunk.usage is None:
                            completion_tokens = string_token_count(all_text)
                            prompt_tokens = string_token_count(
                                str=json.dumps(params))
                            total_tokens = prompt_tokens + completion_tokens
                        else:
                            completion_tokens = chunk.usage.completion_tokens
                            prompt_tokens = chunk.usage.prompt_tokens
                            total_tokens = chunk.usage.total_tokens

                        tps = completion_tokens / (end_time-start_time)
                        logger.info(f"本轮对话{request_id}-tps: {tps}")
                        if tps:
                            model_name = self.model_config["name"]
                            self.profiler_collector(
                                model_name=model_name, key=MODEL_PROF_KEY, value=tps, description="每秒生成token的数量")
                    send_text += text
                    all_text += text
                    if len(send_text) > 2 or finish is True:
                        if finish is True:
                            send_text = send_text.replace("<|", "\n")
                        scheduler.set_result(request_id=request_id,
                                             value=RedisStreamInfer(text=send_text, finish=False, usage=dict(completion_tokens=completion_tokens,
                                                                                                             prompt_tokens=prompt_tokens, total_tokens=total_tokens, tps=tps)))
                        if finish is True:
                            scheduler.set_result(request_id=request_id+"_tool_name",
                                                 value=RedisStreamInfer(
                                                     text=tool_call_function_name, finish=False,
                                                     usage=dict(completion_tokens=completion_tokens,
                                                                prompt_tokens=prompt_tokens, total_tokens=total_tokens, tps=tps)))
                            scheduler.set_result(request_id=request_id+"_tool_args",
                                                 value=RedisStreamInfer(
                                                     text=tool_call_function_args, finish=False,
                                                     usage=dict(completion_tokens=completion_tokens,
                                                                prompt_tokens=prompt_tokens,
                                                                total_tokens=total_tokens, tps=tps)))
                            scheduler.set_result(request_id=request_id,
                                                 value=RedisStreamInfer(text="", finish=True, usage=dict(completion_tokens=completion_tokens,
                                                                                                         prompt_tokens=prompt_tokens, total_tokens=total_tokens, tps=tps)))

                        send_text = ""

        except Exception as e:
            logger.exception(f"推理异常: {e}")
            scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                text="推理服务异常", finish=True))
            # self.current_model = None


class ComfyuiTask(BaseTask):

    def start_comfyui(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        status = True
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            try:
                status = CMD.get_comfyui(device=device, port=port)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_comfyui_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_comfyui, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        # for idx in range(self.workers_num):
        #     self.start_cmd(idx=idx, model_name=model_name)
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def comfyui_infer(self, llm_server, request_id, params, model_config):
        self.update_running_model()
        start_time = time.time()
        data = {"created": 0, "data": []}
        try:
            logger.info(f"处理Comfyui推理任务中: {llm_server} {request_id} {params}")
            if global_config.MOCK:
                data = {"created": 0, "data": [
                    {"url": "http://localhost:8000/images/1.png"}]}
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{json.dumps(data)}",
                                         finish=True))
                time.sleep(1)
            else:
                if self.status == False:
                    raise Exception("模型服务启动异常")
                self.update_running_model()
                self.download_file_to_confyui(
                    download_json_=params.get("download_json"))
                client = ComfyUIClient(server=llm_server, s3_client=s3_client)
                out = client.infer(json_data=params.get(
                    "api_json"), bucket_name=global_config.OSS_CLIENT_CONFIG.get("bucket_name"))
                client.close()

                data = {"created": 0, "data": out}
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{json.dumps(data)}",
                                         finish=True))

                end_time = time.time()

                if len(out) > 0:
                    model_name = self.model_config["name"]
                    self.profiler_collector(
                        model_name=model_name, key=MODEL_PROF_KEY, value=(end_time-start_time)/len(out), description="每张图所耗时间")

        except Exception as e:
            logger.exception(f"推理异常: {e}")
            scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                text=f"{json.dumps(data)}", finish=True))

    def download_file_to_confyui(self, download_json_: dict):
        # 使用wget下载文件到本地，k 是输出的文件名称，v 是下载链接
        for k, v in download_json_.items():
            wget.download(v, k)


class WebuiTask(BaseTask):

    def start_webui(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        status = True
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            try:
                status = CMD.get_webui(device=device, port=port)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_webui_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_webui, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        # for idx in range(self.workers_num):
        #     self.start_cmd(idx=idx, model_name=model_name)
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def webui_infer(self, llm_server, request_id, params, model_config):
        self.update_running_model()
        start_time = time.time()
        data = {"created": 0, "data": []}
        try:
            logger.info(f"处理webui推理任务中: {llm_server} {request_id} {params}")
            if global_config.MOCK:
                data = {"created": 0, "data": [
                    {"url": "http://localhost:8000/images/1.png"}]}
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{json.dumps(data)}",
                                         finish=True))
                time.sleep(1)
            else:
                if self.status == False:
                    raise Exception("模型服务启动异常")
                self.update_running_model()

                client = WebUIClient(server=llm_server, s3_client=s3_client)
                webui_data = SD15MultiControlnetGenerateImage(
                    **params).convert_webui_data()
                out = client.infer(
                    webui_data=webui_data, bucket_name=global_config.OSS_CLIENT_CONFIG.get("bucket_name"))

                data = {"created": 0, "data": out}
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{json.dumps(data)}",
                                         finish=True))
                end_time = time.time()

                if len(out) > 0:
                    model_name = self.model_config["name"]
                    self.profiler_collector(
                        model_name=model_name, key=MODEL_PROF_KEY, value=(end_time-start_time)/len(out), description="每张图所耗时间")

        except Exception as e:
            logger.exception(f"推理异常: {e}")
            scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                text=f"{json.dumps(data)}", finish=True))


class MaskGCTTask(BaseTask):

    def start_maskgct(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        status = True
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            try:
                status = CMD.get_maskgct(device=device, port=port)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_maskgct_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_maskgct, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        # for idx in range(self.workers_num):
        #     self.start_cmd(idx=idx, model_name=model_name)
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def maskgct_infer(self, llm_server, request_id, params, model_config):
        self.update_running_model()
        start_time = time.time()
        try:
            logger.info(f"处理maskgct推理任务中: {llm_server} {request_id} {params}")
            if global_config.MOCK:
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
                                         finish=True))
                time.sleep(1)
            else:
                if self.status == False:
                    raise Exception("模型服务启动异常")
                self.update_running_model()
                client = Client(llm_server)
                clone_url = params.get("clone_url")
                text = params.get("input")
                local_path = client.predict(
                    # filepath in 'Upload Prompt Wav' Audio component
                    file(clone_url),
                    text,  # str in 'Target Text' Textbox component
                    # float in 'Target Duration (in seconds), if the target duration is less than 0, the system will estimate a duration.' Number component
                    -1,
                    25,  # float (numeric value between 15 and 100)
                    api_name="/predict"
                )
                bucket_name = global_config.OSS_CLIENT_CONFIG.get(
                    "bucket_name")
                with open(local_path, "rb") as f:
                    s3_client.upload_fileobj(f, bucket_name, local_path)
                url = s3_client.get_download_url(bucket_name, local_path)

                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{url}",
                                         finish=True))
                end_time = time.time()
                if len(text):
                    model_name = self.model_config["name"]
                    self.profiler_collector(
                        model_name=model_name, key=MODEL_PROF_KEY, value=(end_time-start_time)/len(text), description="每个字符转换语音推理耗时")

        except Exception as e:
            logger.exception(f"推理异常: {e}")
            scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                text="", finish=True))


class FunAsrTask(BaseTask):

    def start_funasr(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            try:
                status = CMD.get_funasr(device=device, port=port)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_funasr_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_funasr, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        # for idx in range(self.workers_num):
        #     self.start_cmd(idx=idx, model_name=model_name)
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def funasr_infer(self, llm_server, request_id, params, model_config):
        self.update_running_model()
        start_time = time.time()
        try:
            logger.info(f"处理funasr推理任务中: {llm_server} {request_id} {params}")
            if global_config.MOCK:
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"测试文本",
                                         finish=True))
                time.sleep(1)
            else:
                if self.status == False:
                    raise Exception("模型服务启动异常")
                self.update_running_model()
                client = Client(llm_server)
                url = params.get("url")
                text = client.predict(
                    # filepath in 'Upload Prompt Wav' Audio component
                    file(url),
                    api_name="/predict"
                )
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{text}",
                                         finish=True))
                end_time = time.time()
                if url:
                    model_name = self.model_config["name"]
                    self.profiler_collector(
                        model_name=model_name, key=MODEL_PROF_KEY, value=(end_time-start_time)/self.get_audio_time(url), description="语音合成：每秒语音的推理时间")

        except Exception as e:
            logger.exception(f"推理异常: {e}")
            scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                text="{}", finish=True))


class EmbeddingTask(BaseTask):

    def start_embedding(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        status = True
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            try:
                status = CMD.get_embedding(device=device, port=port)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_embedding_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_embedding, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        # for idx in range(self.workers_num):
        #     self.start_cmd(idx=idx, model_name=model_name)
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def embedding_infer(self, llm_server, request_id, params, model_config):
        self.update_running_model()
        start_time = time.time()
        try:
            logger.info(f"处理maskgct推理任务中: {llm_server} {request_id} {params}")
            if global_config.MOCK:
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"[0.1, 0.1]",
                                         finish=True))
                time.sleep(1)
            else:
                if self.status == False:
                    raise Exception("模型服务启动异常")
                self.update_running_model()
                client = Client(llm_server)
                input_ = params.get("input", [])
                if type(input_) == str:
                    input_ = [input_]
                model = params.get("model")
                result = client.predict(
                    inputs=input_,
                    model_name=model,
                    api_name="/predict"
                )
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{json.dumps(result)}",
                                         finish=True))
                end_time = time.time()
                if input_:
                    model_name = self.model_config["name"]
                    self.profiler_collector(
                        model_name=model_name, key=MODEL_PROF_KEY, value=(end_time-start_time)/len(input_), description="每个输入转换耗时")

        except Exception as e:
            logger.exception(f"推理异常: {e}")
            scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                text="{}", finish=True))


class LLMTramsformerTask(VllmTask):

    def start_llm_transformer(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        status = True
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            try:
                self.get_chat_template(model_name)
                status = CMD.get_llm_transformer(
                    model_name=model_name, device=device, port=port)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_llm_transformer_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_llm_transformer, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        # for idx in range(self.workers_num):
        #     self.start_cmd(idx=idx, model_name=model_name)
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def llm_transformer_infer(self, llm_server, request_id, params, model_config):
        return self.vllm_infer(llm_server, request_id, params, model_config)


class DiffusersVideoTask(VllmTask):

    def start_diffusers_video(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        status = True
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            try:
                status = CMD.get_diffusers_video(
                    model_name=model_name, device=device, port=port)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_diffusers_video_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_diffusers_video, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        # for idx in range(self.workers_num):
        #     self.start_cmd(idx=idx, model_name=model_name)
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def diffusers_video_infer(self, llm_server, request_id, params, model_config):
        self.update_running_model()
        start_time = time.time()
        try:
            logger.info(
                f"处理diffusers_video推理任务中: {llm_server} {request_id} {params}")
            if global_config.MOCK:
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav",
                                         finish=True))
                time.sleep(1)
            else:
                if self.status == False:
                    raise Exception("模型服务启动异常")
                self.update_running_model()
                ######################################
                model: str = params.get("model")
                prompt: str = params.get("prompt")
                seed: int = params.get("seed")
                width: int = params.get("width")
                height: int = params.get("height")
                num_frames: int = params.get("num_frames", 16)
                ######################################
                client = Client(llm_server)
                result = client.predict(
                    prompt=prompt,
                    negative_prompt="",
                    height=height,
                    width=width,
                    num_inference_steps=50,
                    num_frames=num_frames,
                    guidance_scale=6,
                    seed=42,
                    fps=8,
                    api_name="/infer"
                )
                local_path = result.get("video")
                bucket_name = global_config.OSS_CLIENT_CONFIG.get(
                    "bucket_name")
                with open(local_path, "rb") as f:
                    s3_client.upload_fileobj(f, bucket_name, local_path)
                url = s3_client.get_download_url(bucket_name, local_path)

                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{url}",
                                         finish=True))
                end_time = time.time()
                if len(url):
                    model_name = self.model_config["name"]
                    self.profiler_collector(
                        model_name=model_name, key=MODEL_PROF_KEY, value=(end_time-start_time)/num_frames, description="每帧推理耗时")
        except Exception as e:
            logger.exception(f"推理异常: {e}")
            scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                text="推理异常", finish=True))


class RerankTask(BaseTask):

    def start_rerank(self, idx: int, model_name):
        # 需要保证服务一定完全启动
        status = True
        if global_config.MOCK:
            logger.info(f"本次模拟启动模型: \n{idx} {model_name}")
            time.sleep(5)
        else:
            port = self.worker_start_port + idx
            device = ",".join(map(str, self.split_gpu()[idx]))
            try:
                status = CMD.get_rerank(device=device, port=port)
            except Exception as e:
                status = False
        # self.status = status
        return status

    def start_rerank_server(self, model_name):
        # 启动大模型服务
        self.kill_model_server()  # 要启动就一定要kil旧得进程
        # 改多线程启动模型服务
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            # 提交任务给线程池,线程池大小由实际任务数量决定
            future_to_task = {executor.submit(
                self.start_rerank, i, model_name): i for i in range(self.workers_num)}
            results = [future.result()
                       for future in concurrent.futures.as_completed(future_to_task)]
            if not all(results):
                self.status = False
            else:
                self.status = True
        # for idx in range(self.workers_num):
        #     self.start_cmd(idx=idx, model_name=model_name)
        scheduler.set_running_model(model_name=model_name)
        self.current_model = model_name
        logger.info(f"当前运行模型: {self.current_model} 服务状态: {self.status}")

    def rerank_infer(self, llm_server, request_id, params, model_config):
        self.update_running_model()
        start_time = time.time()
        try:
            logger.info(f"处理rerank推理任务中: {llm_server} {request_id} {params}")
            if global_config.MOCK:
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"[0.1, 0.1]",
                                         finish=True))
                time.sleep(1)
            else:
                if self.status == False:
                    raise Exception("模型服务启动异常")
                self.update_running_model()
                client = Client(llm_server)
                documents = params.get("documents", [])
                if type(documents) == str:
                    input_ = [documents]
                else:
                    input_ = documents
                # inputs: list, query, model_name: str, top_k=3
                result = client.predict(
                    inputs=input_,
                    query=params.get("query"),
                    top_k=params.get("top_n", 3),
                    model_name=params.get("model"),
                    api_name="/predict"
                )
                scheduler.set_result(request_id=request_id,
                                     value=RedisStreamInfer(
                                         text=f"{json.dumps(result)}",
                                         finish=True))
                end_time = time.time()
                if input_:
                    model_name = self.model_config["name"]
                    self.profiler_collector(
                        model_name=model_name, key=MODEL_PROF_KEY, value=(end_time-start_time)/len(input_), description="每个输入的推理时间")

        except Exception as e:
            logger.exception(f"推理异常: {e}")
            scheduler.set_result(request_id=request_id, value=RedisStreamInfer(
                text="{}", finish=True))
            # self.current_model = None
