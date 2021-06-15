Linux Installation
------------------
These installation instructions are valid for FedoraCore 5.  They may work
with or without slight modifications under other Linux distributions.

The Linux installation requires the addition of one file to the directory:

   60-opalkelly.rules ----->  /etc/udev/rules.d/

This file includes a generic udev rule to set the permissions on all
attached Opal Kelly USB devices to allow user access.  Once this file is
in place, you will need to reload the rules by either rebooting or using
the following command:

   /sbin/udevadm control --reload_rules

With these files in place, the Linux device system should automatically
provide write permissions to XEM devices attached to the USB.

