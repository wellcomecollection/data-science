# If you want to create a new Python API running in ECS, copy/paste this
# file into that project.
#
# You'll need to update everything above the divider with the name and
# config of this app, then run `terraform plan` / `terraform apply`.
#
# Terraform will create instructions for deploying code into your new API,
# and how to access it once created.

locals {
  default_tags = {
    TerraformConfigurationURL = "https://github.com/wellcomecollection/data-science/tree/main/semantic-search/terraform"
  }
}

module "api" {
  source = "../../terraform/modules/python_api"

  name = "semantic-search"

  cpu    = 256
  memory = 512

  container_port = 5000

  vpc_id          = local.vpc_id
  private_subnets = local.private_subnets
  public_subnets  = local.public_subnets

  providers = {
    aws.dns = aws.dns
  }
}

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

###############################################################################
# Everything below this line is boilerplate that you don't need to change
# when you copy/paste this file into new projects.
###############################################################################

output "message" {
  value = "Your new API has been created at ${module.api.domain_name}\n\nFor instructions on deploying new code, open ${module.api.readme_path} in your browser"
}

provider "aws" {
  region = "eu-west-1"

  assume_role {
    role_arn = "arn:aws:iam::964279923020:role/data-developer"
  }

  default_tags {
    tags = local.default_tags
  }
}

provider "aws" {
  region = "eu-west-1"
  alias  = "dns"

  assume_role {
    role_arn = "arn:aws:iam::267269328833:role/wellcomecollection-assume_role_hosted_zone_update"
  }

  default_tags {
    tags = local.default_tags
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
  private_subnets = local.data_vpcs["datascience_vpc_private_subnets"]
  public_subnets  = local.data_vpcs["datascience_vpc_public_subnets"]
}
