# MITE: Multi-modal Integrated Training Environment

One paragraph of project description...

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

**Note: All software has been developed and tested for Linux RedHat Fedora 28.**

### Python Prerequisites

This software depends on several Python modules for efficient device interfacing, numerical computation, and data analysis. We suggest installing these packages underneath the local user using the following command in the terminal:

```
python3 -m pip install -U -r requirements.txt --user
```

#### Module Installation Failures

If the following modules fail to install on Fedora 28, install these required packages using the package manager:

**spectrum**
```internet
sudo dnf install python3-devel
sudo dnf install redhat-rpm-config
```

**pybluez**
```
sudo dnf install python3-devel
sudo dnf install bluez-libs-devel
```

### External Device Requirements

The following hardware interfaces require external drivers to be installed:

**Intan RHD2000**
```
sudo cp mite/inputs/external/rhd2000/drivers/Linux/60-opalkelly.rules /etc/udev/rules.d/60-opalkelly.rules; sudo /sbin/udevadm control --reload-rules
```

## TODO
* Add Bayesian filters: ~~Kalman~~, Extended Kalman, Unscented Kalman, Particle