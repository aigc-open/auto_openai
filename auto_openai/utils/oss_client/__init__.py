# coding: utf-8


""""
oss接口
"""
import uuid

import requests
from boto3 import Session
import boto3


class FileManager():
    @staticmethod
    def upload_by_url(url, filename=None, data=None):
        """
        params: url 上传地址
        params: filename 本地文件路径
        params: data file文件对象例如open()等
        """
        if data is None:
            with open(filename, "rb") as f:
                return requests.put(url, data=f)
        else:
            return requests.put(url, data=data)

    @staticmethod
    def download_by_url(self, url, filename):
        """
        params: url 下载地址
        params: filename 本地文件路径
        """
        with open(filename, "wb") as f:
            res = requests.get(url, stream=True)
            for chunk in res.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)


class OSSManager(FileManager):

    def __init__(self, endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None,
                 aws_session_token=None, region_name=None,
                 botocore_session=None, profile_name=None):
        """
        :type aws_access_key_id: string
        :param aws_access_key_id: AWS access key ID
        :type aws_secret_access_key: string
        :param aws_secret_access_key: AWS secret access key
        :type aws_session_token: string
        :param aws_session_token: AWS temporary session token
        :type region_name: string
        :param region_name: Default region when creating new connections
        :type botocore_session: botocore.session.Session
        :param botocore_session: Use this Botocore session instead of creating
                                 a new default one.
        :type profile_name: string
        :param profile_name: The name of a profile to use. If not given, then
                             the default profile is used.
        """
        session = Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            botocore_session=botocore_session,
            profile_name=profile_name
        )
        self.client = session.client('s3', endpoint_url=endpoint_url)
        self.abort_multipart_upload = self.client.abort_multipart_upload
        self.complete_multipart_upload = self.client.complete_multipart_upload
        self.copy_object = self.client.copy_object
        self.create_bucket = self.client.create_bucket
        self.create_multipart_upload = self.client.create_multipart_upload
        self.delete_bucket = self.client.delete_bucket
        self.delete_bucket_analytics_configuration = self.client.delete_bucket_analytics_configuration
        self.delete_bucket_cors = self.client.delete_bucket_cors
        self.delete_bucket_encryption = self.client.delete_bucket_encryption
        self.delete_bucket_intelligent_tiering_configuration = self.client.delete_bucket_intelligent_tiering_configuration
        self.delete_bucket_inventory_configuration = self.client.delete_bucket_inventory_configuration
        self.delete_bucket_lifecycle = self.client.delete_bucket_lifecycle
        self.delete_bucket_metrics_configuration = self.client.delete_bucket_metrics_configuration
        self.delete_bucket_ownership_controls = self.client.delete_bucket_ownership_controls
        self.delete_bucket_policy = self.client.delete_bucket_policy
        self.delete_bucket_replication = self.client.delete_bucket_replication
        self.delete_bucket_tagging = self.client.delete_bucket_tagging
        self.delete_bucket_website = self.client.delete_bucket_website
        self.delete_object = self.client.delete_object
        self.delete_object_tagging = self.client.delete_object_tagging
        self.delete_objects = self.client.delete_objects
        self.delete_public_access_block = self.client.delete_public_access_block
        self.get_bucket_accelerate_configuration = self.client.get_bucket_accelerate_configuration
        self.get_bucket_acl = self.client.get_bucket_acl
        self.get_bucket_analytics_configuration = self.client.get_bucket_analytics_configuration
        self.get_bucket_cors = self.client.get_bucket_cors
        self.get_bucket_encryption = self.client.get_bucket_encryption
        self.get_bucket_intelligent_tiering_configuration = self.client.get_bucket_intelligent_tiering_configuration
        self.get_bucket_inventory_configuration = self.client.get_bucket_inventory_configuration
        self.get_bucket_lifecycle = self.client.get_bucket_lifecycle
        self.get_bucket_lifecycle_configuration = self.client.get_bucket_lifecycle_configuration
        self.get_bucket_location = self.client.get_bucket_location
        self.get_bucket_logging = self.client.get_bucket_logging
        self.get_bucket_metrics_configuration = self.client.get_bucket_metrics_configuration
        self.get_bucket_notification = self.client.get_bucket_notification
        self.get_bucket_notification_configuration = self.client.get_bucket_notification_configuration
        self.get_bucket_ownership_controls = self.client.get_bucket_ownership_controls
        self.get_bucket_policy = self.client.get_bucket_policy
        self.get_bucket_policy_status = self.client.get_bucket_policy_status
        self.get_bucket_replication = self.client.get_bucket_replication
        self.get_bucket_request_payment = self.client.get_bucket_request_payment
        self.get_bucket_tagging = self.client.get_bucket_tagging
        self.get_bucket_versioning = self.client.get_bucket_versioning
        self.get_bucket_website = self.client.get_bucket_website
        self.get_object = self.client.get_object
        self.get_object_acl = self.client.get_object_acl
        self.get_object_legal_hold = self.client.get_object_legal_hold
        self.get_object_lock_configuration = self.client.get_object_lock_configuration
        self.get_object_retention = self.client.get_object_retention
        self.get_object_tagging = self.client.get_object_tagging
        self.get_object_torrent = self.client.get_object_torrent
        self.get_public_access_block = self.client.get_public_access_block
        self.head_bucket = self.client.head_bucket
        self.head_object = self.client.head_object
        self.list_bucket_analytics_configurations = self.client.list_bucket_analytics_configurations
        self.list_bucket_intelligent_tiering_configurations = self.client.list_bucket_intelligent_tiering_configurations
        self.list_bucket_inventory_configurations = self.client.list_bucket_inventory_configurations
        self.list_bucket_metrics_configurations = self.client.list_bucket_metrics_configurations
        self.list_buckets = self.client.list_buckets
        self.list_multipart_uploads = self.client.list_multipart_uploads
        self.list_object_versions = self.client.list_object_versions
        self.list_objects = self.client.list_objects
        self.list_objects_v2 = self.client.list_objects_v2
        self.list_parts = self.client.list_parts
        self.put_bucket_accelerate_configuration = self.client.put_bucket_accelerate_configuration
        self.put_bucket_acl = self.client.put_bucket_acl
        self.put_bucket_analytics_configuration = self.client.put_bucket_analytics_configuration
        self.put_bucket_cors = self.client.put_bucket_cors
        self.put_bucket_encryption = self.client.put_bucket_encryption
        self.put_bucket_intelligent_tiering_configuration = self.client.put_bucket_intelligent_tiering_configuration
        self.put_bucket_inventory_configuration = self.client.put_bucket_inventory_configuration
        self.put_bucket_lifecycle = self.client.put_bucket_lifecycle
        self.put_bucket_lifecycle_configuration = self.client.put_bucket_lifecycle_configuration
        self.put_bucket_logging = self.client.put_bucket_logging
        self.put_bucket_metrics_configuration = self.client.put_bucket_metrics_configuration
        self.put_bucket_notification = self.client.put_bucket_notification
        self.put_bucket_notification_configuration = self.client.put_bucket_notification_configuration
        self.put_bucket_ownership_controls = self.client.put_bucket_ownership_controls
        self.put_bucket_policy = self.client.put_bucket_policy
        self.put_bucket_replication = self.client.put_bucket_replication
        self.put_bucket_request_payment = self.client.put_bucket_request_payment
        self.put_bucket_tagging = self.client.put_bucket_tagging
        self.put_bucket_versioning = self.client.put_bucket_versioning
        self.put_bucket_website = self.client.put_bucket_website
        self.put_object = self.client.put_object
        self.put_object_acl = self.client.put_object_acl
        self.put_object_legal_hold = self.client.put_object_legal_hold
        self.put_object_lock_configuration = self.client.put_object_lock_configuration
        self.put_object_retention = self.client.put_object_retention
        self.put_object_tagging = self.client.put_object_tagging
        self.put_public_access_block = self.client.put_public_access_block
        self.restore_object = self.client.restore_object
        self.select_object_content = self.client.select_object_content
        self.upload_part = self.client.upload_part
        self.upload_part_copy = self.client.upload_part_copy
        self.write_get_object_response = self.client.write_get_object_response
        self.generate_presigned_post = self.client.generate_presigned_post
        self.upload_file = self.client.upload_file
        self.download_file = self.client.download_file
        self.copy = self.client.copy
        self.upload_fileobj = self.client.upload_fileobj
        self.download_fileobj = self.client.download_fileobj
        self.generate_presigned_url = self.client.generate_presigned_url
        self.get_paginator = self.client.get_paginator
        self.can_paginate = self.client.can_paginate
        self.get_waiter = self.client.get_waiter

    def get_chunck_presigned_urls(self, AWS_BUCKET_NAME: str, key: str, chunk_total: int, ExpiresIn=60 * 60 * 24):
        response = self.create_multipart_upload(Bucket=AWS_BUCKET_NAME, Key=key)
        UploadId = response["UploadId"]
        signed_urls = []
        for part_no in range(chunk_total):
            PartNumber = part_no + 1
            signed_url = self.generate_presigned_url(
                ClientMethod='upload_part',
                Params={
                    'Bucket': AWS_BUCKET_NAME,
                    'Key': key,
                    'UploadId': UploadId,
                    'PartNumber': PartNumber
                },
                ExpiresIn=ExpiresIn
            )
            signed_urls.append(signed_url)
        return {"urls": signed_urls, "UploadId": UploadId}

    def merge_ready_chunk(self, AWS_BUCKET_NAME: str, key: str, UploadId: str):
        ready_ = self.list_parts(Bucket=AWS_BUCKET_NAME, Key=key, UploadId=UploadId)
        Parts = []
        for part in ready_["Parts"]:
            Parts.append({
                'PartNumber': part["PartNumber"],
                'ETag': part["ETag"],
            })
        Parts.sort(key=lambda x: x['PartNumber'])
        response = self.complete_multipart_upload(
            Bucket=AWS_BUCKET_NAME,
            Key=key,
            MultipartUpload={'Parts': Parts},
            UploadId=UploadId
        )
        return response

    def get_download_url(self, AWS_BUCKET_NAME, key, ExpiresIn=60 * 60 * 24):
        """
        获取下载素材的地址
        :param AWS_BUCKET_NAME: 桶名称
        :param key: 上传key
        :param ExpiresIn: 过期时间
        :return: url
        """
        url = self.generate_presigned_url("get_object",
                                          Params={'Bucket': AWS_BUCKET_NAME, 'Key': key},
                                          ExpiresIn=ExpiresIn, HttpMethod="GET")

        return url

    def get_upload_url(self, AWS_BUCKET_NAME, key, ExpiresIn=314496000):
        """
        申请上传到oss的地址
        :param AWS_BUCKET_NAME: 桶名称
        :param key: 上传key
        :param ExpiresIn: 过期时间
        :return: url
        """
        url = self.generate_presigned_url("put_object",
                                          Params={'Bucket': AWS_BUCKET_NAME, 'Key': key},
                                          ExpiresIn=ExpiresIn, HttpMethod="PUT")
        return url

    @staticmethod
    def generate_key(directory="", filename=None, is_random=False, is_modify_filename=False):
        """

        :param directory: 存储目录，提供给用户按业务需求分开 例如: test/img
        :param filename: dog。jpg
        :param is_random: 是否生成随机key
        :param is_modify_filename: 如果随机是否修改文件名
        :return: 例如：
            test/img/aab57cbb-b319-4c72-bdba-e699d71a4ebf/dog.jpg 这种文件下载下来保持原文件，且不会重名
            test/img/aab57cbb-b319-4c72-bdba-e699d71a4ebf.jpg 这种文件下载下来名称会被修改，且不会重名
        """
        if is_random:
            if is_modify_filename is False:
                return directory + "/" + str(uuid.uuid4()) + "/" + filename.split("/")[-1]
            else:
                return directory + "/" + str(uuid.uuid4()) + "." + filename.split(".")[-1]
        else:
            return directory + "/" + filename.split("/")[-1]


class OSSSTS(FileManager):

    def __init__(self, endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None,
                 aws_session_token=None, region_name=None,
                 botocore_session=None, profile_name=None):
        """
        :type aws_access_key_id: string
        :param aws_access_key_id: AWS access key ID
        :type aws_secret_access_key: string
        :param aws_secret_access_key: AWS secret access key
        :type aws_session_token: string
        :param aws_session_token: AWS temporary session token
        :type region_name: string
        :param region_name: Default region when creating new connections
        :type botocore_session: botocore.session.Session
        :param botocore_session: Use this Botocore session instead of creating
                                 a new default one.
        :type profile_name: string
        :param profile_name: The name of a profile to use. If not given, then
                             the default profile is used.
        """

        self.client = boto3.client('sts', endpoint_url=endpoint_url, aws_access_key_id=aws_access_key_id,
                                   aws_secret_access_key=aws_secret_access_key,
                                   aws_session_token=aws_session_token,
                                   region_name=region_name)
