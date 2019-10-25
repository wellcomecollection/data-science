terraform {
  required_version = ">= 0.11"

  backend "s3" {
    role_arn = "arn:aws:iam::964279923020:role/data-developer"

    bucket         = "wellcomecollection-datascience-infra"
    key            = "terraform/datascience_data.tfstate"
    dynamodb_table = "terraform-locktable"
    region         = "eu-west-1"
  }
}

provider "aws" {
  assume_role {
    role_arn = "arn:aws:iam::964279923020:role/data-admin"
  }

  region  = "eu-west-1"
  version = "1.59.0"
}
