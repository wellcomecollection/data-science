terraform {
  required_version = ">= 0.11"

  backend "s3" {
    role_arn = "arn:aws:iam::964279923020:role/data-developer"

    bucket         = "wellcomecollection-datascience-infra"
    key            = "terraform/labs_apps.tfstate"
    dynamodb_table = "terraform-locktable"
    region         = "eu-west-1"
  }
}

data "terraform_remote_state" "accounts_data" {
  backend = "s3"

  config {
    role_arn = "arn:aws:iam::760097843905:role/platform-developer"

    bucket = "wellcomecollection-platform-infra"
    key    = "terraform/accounts/data.tfstate"
    region = "eu-west-1"
  }
}

provider "aws" {
  assume_role {
    role_arn = "arn:aws:iam::964279923020:role/data-admin"
  }

  region = "eu-west-1"
}

provider "template" {
  version = "~> 2.1"
}
