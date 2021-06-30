output "name" {
  value = aws_ecs_service.service.name
}

output "target_group_arn" {
  value = aws_lb_target_group.tcp.arn
}
