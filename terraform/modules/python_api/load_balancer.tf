data "aws_vpc" "vpc" {
  id = var.vpc_id
}

resource "aws_security_group" "load_balancer" {
  name        = "${var.name}-load_balancer"
  description = "Allow traffic between services and the load balancer"
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
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.vpc.cidr_block]
  }
}

resource "aws_security_group" "interservice" {
  name        = "${var.name}-interservice"
  vpc_id      = var.vpc_id

  ingress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"
    self      = true
  }
}

resource "aws_security_group" "allow_all_ingress" {
  name        = "${var.name}-all_ingress"
  description = "Allow traffic from anywhere on the Internet"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "allow_all_egress" {
  name        = "${var.name}-all_egress"
  description = "Allow traffic to anywhere on the Internet"
  vpc_id      = var.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_alb" "alb" {
  name            = var.name
  subnets         = var.subnets
  security_groups = [aws_security_group.load_balancer.id, aws_security_group.interservice.id, aws_security_group.allow_all_ingress.id, aws_security_group.allow_all_egress.id]
}

resource "aws_alb_target_group" "ecs_service_default" {
  name = "${var.name}-default"
  port = 80
  protocol = "HTTP"
  vpc_id = var.vpc_id
}

resource "aws_alb_listener" "http" {
  load_balancer_arn = aws_alb.alb.id
  port = "80"
  protocol = "HTTP"

  default_action {
    target_group_arn = aws_lb_target_group.http.arn
    type = "forward"
  }
}

resource "aws_lb_listener_rule" "http" {
  listener_arn = aws_alb_listener.http.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.http.arn
  }

  condition {
    path_pattern {
      values = ["/*"]
    }
  }
}
