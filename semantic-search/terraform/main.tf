locals {
  default_tags = {
    TerraformConfigurationURL = "https://github.com/wellcomecollection/data-science/tree/main/semantic-search/terraform"
  }
}

module "api" {
  source = "../../terraform/modules/python_api"

  name = "semantic-search"

  cpu = 256
  memory = 512

  container_port = 5000

  vpc_id = local.vpc_id
  subnets = local.subnets
}
