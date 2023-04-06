resource "aws_security_group" "lb_ingress" {
  name        = "${var.name}-service_lb_ingress"
  description = "Allow traffic from the ALB into services"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    security_groups = [aws_security_group.lb.id]
  }

  tags = {
    Name = "${var.name}-lb-ingress"
  }
}

resource "aws_security_group" "egress" {
  name        = "${var.name}-service_egress"
  description = "Allows all egress traffic from the group"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name}-service_egress"
  }
}

resource "aws_security_group" "lb" {
  name        = "${var.name}-lb"
  description = "Allow traffic between services and ALB"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = module.service.nginx_container_port
    to_port     = module.service.nginx_container_port
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.vpc.cidr_block]
  }

  tags = {
    Name = "${var.name}-lb"
  }
}
