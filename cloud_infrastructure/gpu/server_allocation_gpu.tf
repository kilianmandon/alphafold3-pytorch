terraform {
  required_providers {
    digitalocean = {
      source = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

variable "do_token" {
    sensitive = true
}

provider "digitalocean" {
    token = var.do_token
}

data "digitalocean_ssh_key" "terraform" {
  name = "digital-ocean"
}

resource "digitalocean_droplet" "af3_training" {
  image = "gpu-h100x1-base"
  name = "af3_training"
  region = "tor1"
  size = "gpu-4000adax1-20gb"
  ssh_keys = [
    data.digitalocean_ssh_key.terraform.id
  ]
  user_data = file("cloud-init.sh")

}

output "instance_ips" {
    value = digitalocean_droplet.af3_training.ipv4_address
}