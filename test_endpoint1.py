import argparse
import json
import boto3


def test_endpoint(endpoint_name: str, region: str = None):
    # Automatically detects the region from your AWS environment if not provided
    session = boto3.Session(region_name=region)
    runtime = session.client("sagemaker-runtime")

    test_inputs = [
        "The company reported record profits and strong revenue growth.",
        "The stock plummeted after missing earnings expectations.",
        "Trading volume remained steady with no major changes.",
    ]

    print(f"Sending test request to endpoint: {endpoint_name}...")

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"inputs": test_inputs}),
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    print("\nPredictions:")
    for text, pred in zip(test_inputs, result):
        # Handles both numeric string labels (e.g., "2") and text labels (e.g., "positive")
        raw_label = str(pred["label"]).replace("LABEL_", "")
        label = (
            id2label.get(int(raw_label), raw_label)
            if raw_label.isdigit()
            else raw_label
        )
        print(
            f"  [Sentiment: {label:8} - Confidence: {pred['score']:.2f}]  {text[:60]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test deployed FinBERT SageMaker endpoint"
    )
    parser.add_argument(
        "--endpoint-name",
        default="finbert-sentiment-endpoint",
        help="Name of the deployed SageMaker endpoint",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (defaults to your current environment region)",
    )
    args = parser.parse_args()

    test_endpoint(args.endpoint_name, args.region)
