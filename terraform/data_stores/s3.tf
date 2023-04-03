resource "aws_s3_bucket" "miro_images_feature_vectors" {
  bucket = "miro-images-feature-vectors"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_acl" "miro_images_feature_vectors" {
  bucket = aws_s3_bucket.miro_images_feature_vectors.id
  acl    = "private"
}

resource "aws_s3_bucket" "model_core_data" {
  bucket = "model-core-data"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_acl" "model_core_data" {
  bucket = aws_s3_bucket.model_core_data.id
  acl    = "private"
}

# policies
resource "aws_s3_bucket_policy" "model_core_data_policy" {
  bucket = aws_s3_bucket.model_core_data.bucket
  policy = data.aws_iam_policy_document.s3_read.json
}

data "aws_iam_policy_document" "s3_read" {
  statement {
    actions = ["s3:*"]

    resources = [
      aws_s3_bucket.model_core_data.arn,
      "${aws_s3_bucket.model_core_data.arn}/*",
    ]

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::964279923020:role/datascience_ec2"]
    }
  }
}
