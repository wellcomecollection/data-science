terraform {
  required_version = ">= 0.9"

  backend "s3" {
    bucket         = "wellcomecollection-platform-infra"
    key            = "terraform/data-science/semantic-search.tfstate"
    dynamodb_table = "terraform-locktable"

    role_arn = "arn:aws:iam::760097843905:role/platform-developer"
    region   = "eu-west-1"
  }
}

data "terraform_remote_state" "accounts_data" {
  backend = "s3"

  config = {
    role_arn = "arn:aws:iam::760097843905:role/platform-read_only"
    bucket   = "wellcomecollection-platform-infra"
    key      = "terraform/platform-infrastructure/accounts/data.tfstate"
    region   = "eu-west-1"
  }
}

locals {
  data_vpcs = data.terraform_remote_state.accounts_data.outputs

  vpc_id          = local.data_vpcs["datascience_vpc_id"]
  subnets         = local.data_vpcs["datascience_vpc_public_subnets"]
}
