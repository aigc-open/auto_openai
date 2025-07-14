#!/bin/bash

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# 默认配置参数（如果配置文件中没有设置）
MODEL_NAME="${MODEL_NAME:-Qwen2.5-Coder-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/root/share_models/LLM/Qwen2.5-Coder-7B-Instruct/}"
OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:9000}"
WAIT_TIME="${WAIT_TIME:-10}"
LOG_LEVEL="${LOG_LEVEL:-info}"
VERBOSE="${VERBOSE:-false}"
FORCE_RECREATE="${FORCE_RECREATE:-false}"

# 日志函数
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "[$timestamp] [$level] $message"
    
    if [[ "$VERBOSE" == "true" ]]; then
        echo "[$timestamp] [$level] $message" >&2
    fi
}

# 函数：打印使用说明
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --model-name    Model name (default: $MODEL_NAME)"
    echo "  -p, --model-path    Model path (default: $MODEL_PATH)"
    echo "  -h, --host          Host address (default: $OLLAMA_HOST)"
    echo "  -w, --wait-time     Wait time in seconds (default: $WAIT_TIME)"
    echo "  -v, --verbose       Enable verbose output"
    echo "  -f, --force-recreate Force recreate model even if it exists"
    echo "  --help              Show this help message"
    echo ""
}

# 函数：解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            -p|--model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            -h|--host)
                OLLAMA_HOST="$2"
                shift 2
                ;;
            -w|--wait-time)
                WAIT_TIME="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -f|--force-recreate)
                FORCE_RECREATE="true"
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# 函数：检查必要的依赖
check_dependencies() {
    log "INFO" "Checking dependencies..."
    
    if ! command -v ollama &> /dev/null; then
        log "ERROR" "ollama is not installed or not in PATH"
        exit 1
    fi
    
    if [[ ! -d "$MODEL_PATH" ]]; then
        log "ERROR" "Model path does not exist: $MODEL_PATH"
        exit 1
    fi
    
    log "INFO" "Dependencies check passed"
}

# 函数：停止现有的ollama进程
stop_ollama() {
    log "INFO" "Stopping existing ollama processes..."
    local pids=$(ps -ef | grep ollama | grep -v grep | awk '{print $2}')
    
    if [[ -n "$pids" ]]; then
        echo "$pids" | xargs -r kill -9
        log "INFO" "Stopped ollama processes: $pids"
    else
        log "INFO" "No existing ollama processes found"
    fi
    
    sleep 2
}

# 函数：启动ollama服务
start_ollama_service() {
    log "INFO" "Starting ollama service..."
    
    # 清理可能存在的锁文件
    rm -f /tmp/ollama.lock
    
    ollama serve &
    local pid=$!
    
    log "INFO" "Ollama service started with PID: $pid"
    log "INFO" "Waiting ${WAIT_TIME}s for service to initialize..."
    
    sleep $WAIT_TIME
    
    # 检查服务是否正常运行
    if ! ps -p $pid > /dev/null 2>&1; then
        log "ERROR" "Ollama service failed to start"
        exit 1
    fi
    
    log "INFO" "Ollama service initialization completed"
}

# 函数：删除已存在的模型
remove_existing_model() {
    local model_name="$1"
    log "INFO" "Removing existing model: $model_name"
    
    if ollama rm "$model_name" 2>/dev/null; then
        log "INFO" "Successfully removed existing model: $model_name"
    else
        log "WARN" "Failed to remove existing model: $model_name (it may not exist)"
    fi
}

# 函数：检查模型是否已存在
check_model_exists() {
    local model_name="$1"
    log "INFO" "Checking if model '$model_name' already exists..."
    
    # 获取ollama模型列表
    local model_list=$(ollama list 2>/dev/null)
    
    if [[ $? -ne 0 ]]; then
        log "WARN" "Failed to get model list, will proceed with model creation"
        return 1
    fi
    
    # 检查模型是否存在（忽略大小写）
    if echo "$model_list" | grep -i "^$model_name" > /dev/null; then
        log "INFO" "Model '$model_name' already exists"
        return 0
    else
        log "INFO" "Model '$model_name' does not exist, will create it"
        return 1
    fi
}

# 函数：创建模型
create_model() {
    log "INFO" "Creating model: $MODEL_NAME"
    log "INFO" "Using model path: $MODEL_PATH"
    log "INFO" "Force recreate: $FORCE_RECREATE"
    
    # 检查模型是否已存在
    if check_model_exists "$MODEL_NAME"; then
        if [[ "$FORCE_RECREATE" == "true" ]]; then
            log "INFO" "Force recreate is enabled, removing existing model"
            remove_existing_model "$MODEL_NAME"
        else
            log "INFO" "Skipping model creation as model already exists (use -f to force recreate)"
            return 0
        fi
    fi
    
    # 创建Modelfile
    local modelfile_content="FROM $MODEL_PATH"
    echo "$modelfile_content" > Modelfile
    
    log "INFO" "Created Modelfile with content: $modelfile_content"
    
    # 创建模型
    if ollama create "$MODEL_NAME" -f Modelfile; then
        log "INFO" "Model $MODEL_NAME created successfully"
    else
        log "ERROR" "Failed to create model $MODEL_NAME"
        exit 1
    fi
    
    # 清理临时文件
    rm -f Modelfile
}

# 函数：重新启动ollama服务
restart_ollama_service() {
    log "INFO" "Restarting ollama service with host: $OLLAMA_HOST"
    
    stop_ollama
    
    export OLLAMA_HOST=$OLLAMA_HOST
    log "INFO" "Set OLLAMA_HOST to: $OLLAMA_HOST"
    
    # 启动服务
    ollama serve
}

# 函数：验证服务状态
verify_service() {
    log "INFO" "Verifying service status..."
    
    # 尝试连接服务
    local max_attempts=5
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "http://$OLLAMA_HOST/api/version" > /dev/null 2>&1; then
            log "INFO" "Service is responding correctly"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log "WARN" "Service not responding, attempt $attempt/$max_attempts"
        sleep 2
    done
    
    log "ERROR" "Service verification failed after $max_attempts attempts"
    return 1
}

# 主函数
main() {
    log "INFO" "=== Ollama Setup Script Started ==="
    log "INFO" "Model Name: $MODEL_NAME"
    log "INFO" "Model Path: $MODEL_PATH"
    log "INFO" "Host: $OLLAMA_HOST"
    log "INFO" "Wait Time: ${WAIT_TIME}s"
    log "INFO" "Force Recreate: $FORCE_RECREATE"
    log "INFO" "======================================="
    
    check_dependencies
    start_ollama_service
    create_model
    restart_ollama_service
    verify_service
    
    log "INFO" "=== Ollama Setup Completed Successfully ==="
}

# 错误处理
trap 'log "ERROR" "Script interrupted"; exit 1' INT TERM

# 解析命令行参数
parse_args "$@"

# 运行主函数
main

# ./start.sh --model-name "Qwen2.5-Coder-1.5B-Instruct" --model-path "/root/share_models/LLM/Qwen2.5-Coder-1.5B-Instruct/" --host "0.0.0.0:8000"