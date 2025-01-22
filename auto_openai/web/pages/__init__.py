import pandas as pd
from typing import Dict, Any
web_prefix = ""


def format_constraints(constraints):
    def format_constraint(constraint):
        constraint_strs = []
        if constraint.get("type") == "integer":
            constraint_strs.append("类型: integer")
        elif constraint.get("type") == "boolean":
            constraint_strs.append("类型: boolean")
        elif constraint.get("type") == "string":
            constraint_strs.append("类型: string")
        elif constraint.get("type") == "array":
            constraint_strs.append(
                f"类型: array")
        elif constraint.get("type") == "null":
            constraint_strs.append("类型: null")
        else:
            constraint_strs.append(f"类型: {constraint.get('type')}")

        if "exclusiveMaximum" in constraint and "exclusiveMinimum" in constraint:
            constraint_strs.append(
                f"约束: {constraint['exclusiveMinimum']} < value < {constraint['exclusiveMaximum']}")
        elif "maximum" in constraint and "minimum" in constraint:
            constraint_strs.append(
                f"约束: {constraint['minimum']} ≤ value ≤ {constraint['maximum']}")

        return constraint_strs
    if not constraints:
        return ""
    constraint_strs = []
    if type(constraints) == list:
        for constraint in constraints:
            constraint_strs.extend(format_constraint(constraint))
    elif type(constraints) == dict:
        constraint_strs = format_constraint(constraints)
    return " 或 ".join(constraint_strs)


def generate_api_documentation(schema: Dict[str, Any]):
    # 创建一个空列表来存储每个属性的信息
    data = []
    # 处理属性
    properties = schema.get('properties', {})
    for prop_name, prop_details in properties.items():
        title = prop_details.get('title', prop_name)
        default = prop_details.get('default', '无')
        prop_type = format_constraints(prop_details.get("anyOf"))
        prop_type = prop_type if prop_type else format_constraints(
            prop_details)
        description = prop_details.get('description', '无描述')
        # 处理枚举类型
        if '$ref' in prop_details:
            ref = prop_details['$ref']
            # 动态查找枚举值
            enum_values = schema['$defs'].get(
                ref.split('/')[-1], {}).get('enum', [])
            if enum_values:
                description = f" (可选值: {', '.join(enum_values)})"
                prop_type = "string"  # 假设枚举类型为字符串
        elif 'allOf' in prop_details:
            for item in prop_details['allOf']:
                if '$ref' in item:
                    ref = item['$ref']
                    # 动态查找枚举值
                    enum_values = schema['$defs'].get(
                        ref.split('/')[-1], {}).get('enum', [])
                    if enum_values:
                        description = f" (可选值: {', '.join(enum_values)})"
                        prop_type = "string"  # 假设枚举类型为字符串

        current_data = {
            '名称': prop_name,
            '约束/类型': prop_type,
            '默认值/参考': f'{default}',
            '描述': description
        }
        # 将属性信息添加到列表中
        data.append(current_data)
    df = pd.DataFrame(data)
    return df
