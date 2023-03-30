provider "aws" {
  region = "eu-west-1"

  assume_role {
    role_arn = "arn:aws:iam::964279923020:role/data-developer"
  }

  default_tags {
    tags = local.default_tags
  }
}
