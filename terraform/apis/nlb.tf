module "nlb" {
  source = "git::https://github.com/wellcometrust/terraform.git//load_balancer/network?ref=v19.6.3"

  namespace       = "datascience-apis"
  private_subnets = ["${local.private_subnets}"]
}
