resource "aws_ecr_repository" "api" {
  name = "weco/${var.name}"
}
