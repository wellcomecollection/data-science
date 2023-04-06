resource "aws_ecr_repository" "api" {
  name = "weco/${var.name}"

  # This will delete the ECR repository when you run `terraform destroy`.
  #
  # In general we don't want this because it's potentially disruptive,
  # but since this is only used for short-lived experiments and
  # ephemeral apps, it's fine.
  force_delete = true
}
