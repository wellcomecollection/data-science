locals {
  nginx_container_image          = "wellcome/nginx_apigw:8c68706fba97189fc72e3d0844458c7c5dee0bbc"
  nginx_listener_port            = "9000"
  public_subnets                 = "${data.terraform_remote_state.accounts_data.datascience_vpc_public_subnets}"
  private_subnets                = "${data.terraform_remote_state.accounts_data.datascience_vpc_private_subnets}"
  vpc_id                         = "${data.terraform_remote_state.accounts_data.datascience_vpc_id}"
  namespace_id                = "datascience"
  cluster_id                  = "apis"
  miro_read_role              = "arn:aws:iam::760097843905:role/sourcedata-miro-assumable_read_role"
}

data "aws_vpc" "vpc" {
  id = "${local.vpc_id}"
}
