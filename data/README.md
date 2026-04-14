# Sample data manifests for each task.
# Place your actual images under data/images/ and reference this dir.
# Each file must be named <task_name>.json and contain a JSON array.

# captioning.json format:
# [{"image": "images/dog.jpg", "prompt": "Describe this image.", "reference": "A golden dog..."}]

# vqa.json format:
# [{"image": "images/chart.png", "prompt": "What is the value at index 3?", "reference": "42"}]

# structured_output.json format:
# [{
#   "image": "images/receipt.jpg",
#   "prompt": "Extract invoice data from this receipt.",
#   "reference": "{\"total\": \"12.99\", \"vendor\": \"Acme\"}",
#   "metadata": {
#     "schema": "{\"type\":\"object\",\"properties\":{\"total\":{\"type\":\"string\"},\"vendor\":{\"type\":\"string\"}}}"
#   }
# }]
