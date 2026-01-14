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

resource "digitalocean_droplet" "af3-setup" {
  image = "ubuntu-22-04-x64"
  name = "af3-setup"
  region = "tor1"
  size = "s-4vcpu-8gb-240gb-intel"
  ssh_keys = [
    data.digitalocean_ssh_key.terraform.id
  ]
  user_data = file("cpu_server_cloud_init.sh")

}

output "instance_ips" {
    value = digitalocean_droplet.af3-setup.ipv4_address
}