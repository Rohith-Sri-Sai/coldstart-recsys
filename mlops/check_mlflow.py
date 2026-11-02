import os
import subprocess

import dagshub
import mlflow


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main():
    owner = os.environ["DAGSHUB_REPO_OWNER"]
    repo = os.environ["DAGSHUB_REPO_NAME"]

    dagshub.init(
        repo_owner=owner,
        repo_name=repo,
        mlflow=True,
        dvc=False,
    )

    mlflow.set_experiment("recsys-integration-check")

    with mlflow.start_run(run_name="first-connection"):
        mlflow.log_param("project", "coldstart-recsys")
        mlflow.log_param("dataset", "all_beauty")

        # Connectivity check, not a model-quality result.
        mlflow.log_metric("connection_success", 1.0)

        mlflow.set_tag("git_commit", git_sha())
        mlflow.set_tag("run_type", "integration-test")

        run_id = mlflow.active_run().info.run_id

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("MLflow run ID:", run_id)


if __name__ == "__main__":
    main()
