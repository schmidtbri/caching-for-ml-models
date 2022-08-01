# Terraform 

This example will:

- create a docker registry
- create a VPC
- create a kubernetes cluster with 2 nodes (the default number of nodes)
- create an additional node pool in the cluster

## Pre-requisites

* You must have [Terraform](https://www.terraform.io/) installed on your computer. 
* You must have an DigitalOcean account.
* You must have a personal access token.

## Quick start

**Please note that this example will deploy real resources into your account, and we are not responsible for any
charges you may incur.** 

Configure your [Digital Ocean Token](https://docs.digitalocean.com/reference/api/create-personal-access-token/) as an 
environment variable:

```bash
export DIGITALOCEAN_TOKEN=(your access token)
```

View the plan:
```bash
terraform init
terraform plan
```

Deploy the code:
```bash
terraform apply
```

Clean up when you're done:

```bash
terraform destroy
```
