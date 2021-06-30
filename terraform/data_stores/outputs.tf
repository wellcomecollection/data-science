output "miro_images_feature_vectors_bucket_id" {
  value = aws_s3_bucket.miro_images_feature_vectors.id
}

output "miro_images_feature_vectors_bucket_arn" {
  value = aws_s3_bucket.miro_images_feature_vectors.arn
}

output "model_core_data_bucket_id" {
  value = aws_s3_bucket.model_core_data.id
}

output "model_core_data_bucket_arn" {
  value = aws_s3_bucket.model_core_data.arn
}
