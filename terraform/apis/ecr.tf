resource "aws_ecr_repository" "feature_similarity" {
  name = "weco/feature_similarity"
}

resource "aws_ecr_repository" "palette_similarity" {
  name = "weco/palette_similarity"
}

resource "aws_ecr_repository" "nginx" {
  name = "weco/nginx"
}
