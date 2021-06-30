module "nlb" {
  source = "./modules/load_balancer/network"

  namespace       = "datascience-apis"
  private_subnets = ["${local.private_subnets}"]
}
