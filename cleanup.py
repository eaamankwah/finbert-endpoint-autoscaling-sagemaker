import argparse
import boto3
from botocore.exceptions import ClientError


def delete_s3_resources(
    s3_resource,
    bucket_name: str,
    prefix: str = None,
    delete_bucket: bool = False,
):
    bucket = s3_resource.Bucket(bucket_name)
    try:
        if delete_bucket:
            # Deletes all object versions and delete markers (works for versioned and non-versioned buckets)
            bucket.object_versions.all().delete()
            bucket.objects.all().delete()
            bucket.delete()
            print(f"[Deleted] S3 Bucket and all contents: s3://{bucket_name}")
        elif prefix:
            clean_prefix = prefix.lstrip("/")
            bucket.object_versions.filter(Prefix=clean_prefix).delete()
            bucket.objects.filter(Prefix=clean_prefix).delete()
            print(
                f"[Deleted] S3 objects under: s3://{bucket_name}/{clean_prefix}"
            )
        else:
            print(
                "[Skipped] Provide --s3-prefix to delete specific files or --delete-bucket to remove the whole bucket."
            )
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            print(f"[Skipped] S3 Bucket '{bucket_name}' does not exist.")
        else:
            print(
                f"[Warning] Error cleaning up S3 bucket '{bucket_name}': {e}"
            )


def cleanup_model_registry(sm_client, package_group_name: str):
    try:
        # 1. List and delete all model package versions inside the group first
        paginator = sm_client.get_paginator("list_model_packages")
        for page in paginator.paginate(
            ModelPackageGroupName=package_group_name
        ):
            for pkg in page.get("ModelPackageSummaryList", []):
                pkg_arn = pkg["ModelPackageArn"]
                sm_client.delete_model_package(ModelPackageName=pkg_arn)
                print(f"[Deleted] Model Package Version: {pkg_arn}")

        # 2. Delete the Model Package Group itself
        sm_client.delete_model_package_group(
            ModelPackageGroupName=package_group_name
        )
        print(f"[Deleted] Model Package Group: {package_group_name}")
    except ClientError as e:
        if (
            "does not exist" in str(e)
            or e.response["Error"]["Code"] == "ResourceNotFound"
        ):
            print(
                f"[Skipped] Model Package Group '{package_group_name}' does not exist."
            )
        else:
            print(f"[Warning] Error cleaning up Model Registry: {e}")


def cleanup_resources(
    endpoint_name: str,
    package_group_name: str = None,
    pipeline_name: str = None,
    bucket_name: str = None,
    s3_prefix: str = None,
    delete_bucket: bool = False,
    region: str = None,
):
    session = boto3.Session(region_name=region)
    sm_client = session.client("sagemaker")
    aas_client = session.client("application-autoscaling")
    s3_resource = session.resource("s3")
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    print(f"Starting full cleanup in region '{session.region_name}'...\n")

    # 1. Remove Auto Scaling Policies and Deregister Scalable Target
    try:
        policies = aas_client.describe_scaling_policies(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        ).get("ScalingPolicies", [])

        for policy in policies:
            policy_name = policy["PolicyName"]
            aas_client.delete_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            )
            print(f"[Deleted] Auto Scaling policy: {policy_name}")

        aas_client.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
        print(f"[Deleted] Auto Scaling scalable target: {resource_id}")
    except ClientError as e:
        if e.response["Error"]["Code"] in [
            "ObjectNotFoundException",
            "ValidationException",
        ]:
            print("[Skipped] No active Auto Scaling target found.")
        else:
            print(f"[Warning] Error cleaning up Auto Scaling: {e}")

    # 2. Retrieve Endpoint Config and Model Name(s) before deleting the Endpoint
    endpoint_config_name = None
    model_names = []

    try:
        endpoint_desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_desc.get("EndpointConfigName")
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            print(f"[Skipped] Endpoint '{endpoint_name}' does not exist.")
        else:
            print(f"[Warning] Error describing endpoint: {e}")

    if endpoint_config_name:
        try:
            config_desc = sm_client.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            for variant in config_desc.get("ProductionVariants", []):
                if "ModelName" in variant:
                    model_names.append(variant["ModelName"])
        except ClientError as e:
            print(f"[Warning] Error describing endpoint config: {e}")

    # 3. Delete the SageMaker Endpoint (Stops hourly compute billing immediately)
    try:
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"[Deleted] SageMaker Endpoint: {endpoint_name}")
    except ClientError as e:
        if "Could not find endpoint" not in str(e):
            print(f"[Warning] Error deleting endpoint: {e}")

    # 4. Delete the Endpoint Configuration
    if endpoint_config_name:
        try:
            sm_client.delete_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            print(
                f"[Deleted] SageMaker Endpoint Config: {endpoint_config_name}"
            )
        except ClientError as e:
            print(f"[Warning] Error deleting endpoint config: {e}")

    # 5. Delete the SageMaker Model(s)
    for model_name in set(model_names):
        try:
            sm_client.delete_model(ModelName=model_name)
            print(f"[Deleted] SageMaker Model: {model_name}")
        except ClientError as e:
            print(f"[Warning] Error deleting model '{model_name}': {e}")

    # 6. Delete Model Registry Package Group and Versions
    if package_group_name:
        cleanup_model_registry(sm_client, package_group_name)

    # 7. Delete SageMaker Pipeline (Optional)
    if pipeline_name:
        try:
            sm_client.delete_pipeline(PipelineName=pipeline_name)
            print(f"[Deleted] SageMaker Pipeline: {pipeline_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFound":
                print(f"[Skipped] Pipeline '{pipeline_name}' does not exist.")
            else:
                print(f"[Warning] Error deleting pipeline: {e}")

    # 8. Clean Up S3 Artifacts / Bucket
    if bucket_name:
        delete_s3_resources(s3_resource, bucket_name, s3_prefix, delete_bucket)

    print("\nFull infrastructure cleanup complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tear down SageMaker endpoint, Model Registry, Pipeline, and S3 resources"
    )
    parser.add_argument(
        "--endpoint-name",
        default="finbert-sentiment-endpoint",
        help="Name of the deployed SageMaker endpoint",
    )
    parser.add_argument(
        "--package-group",
        default=None,
        help="Name of the SageMaker Model Registry Package Group to delete",
    )
    parser.add_argument(
        "--pipeline-name",
        default=None,
        help="Name of the SageMaker Pipeline to delete",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Name of the S3 bucket containing pipeline/model artifacts",
    )
    parser.add_argument(
        "--s3-prefix",
        default="finbert/",
        help="S3 folder prefix to delete (default: 'finbert/'). Ignored if --delete-bucket is set.",
    )
    parser.add_argument(
        "--delete-bucket",
        action="store_true",
        help="Empty and permanently delete the entire S3 bucket",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (defaults to your current environment region)",
    )
    args = parser.parse_args()

    cleanup_resources(
        endpoint_name=args.endpoint_name,
        package_group_name=args.package_group,
        pipeline_name=args.pipeline_name,
        bucket_name=args.bucket,
        s3_prefix=args.s3_prefix,
        delete_bucket=args.delete_bucket,
        region=args.region,
    )
