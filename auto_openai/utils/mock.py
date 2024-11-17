
LLM_MOCK_DATA = """
# Ubuntu 22.04 LTS 降级内核版本记录

## 背景

在使用Ubuntu 22.04 LTS的过程中，我遇到了一些虚拟机（VM）无故当掉的问题。经过调查，我决定将内核版本从5.15.0降级到5.13.0，以期解决这些问题。

## 降级步骤

### 1. 下载内核管理脚本

首先，下载用于管理内核的Bash脚本：

`wget https://raw.githubusercontent.com/pimlie/ubuntu-mainline-kernel.sh/master/ubuntu-mainline-kernel.sh`

### 2. 赋予执行权限并移动脚本

接下来，赋予脚本执行权限并将其移动到本地bin目录：

`chmod +x ubuntu-mainline-kernel.sh`  
`sudo mv ubuntu-mainline-kernel.sh /usr/local/bin/`

### 3. 检查可用内核版本

使用以下命令列出可用的内核版本：

`ubuntu-mainline-kernel.sh -r`

### 4. 安装指定版本的内核

安装5.13.0版本的内核：

`sudo ubuntu-mainline-kernel.sh -i v5.13.0`

### 5. 更新GRUB配置

修改GRUB配置以设置默认启动内核：

`sudo nano /etc/default/grub`

将`GRUB_DEFAULT`的值修改为：
GRUB_DEFAULT='Advanced options for Ubuntu>Ubuntu, with Linux 5.13.0-051300-generic'



然后更新GRUB：

`sudo update-grub`

### 6. 重启系统

重启系统以应用更改：

`sudo reboot`

### 7. 验证内核版本

登录后，使用以下命令确认当前内核版本：

`uname -r`
## 参考链接

- [How to Change Default Kernel in Ubuntu](https://www.how2shout.com/linux/how-to-change-default-kernel-in-ubuntu-22-04-20-04-lts/#6_To_install_any_specific_or_old_version)
- [Ask Ubuntu: Downgrade Kernel for Ubuntu 22.04 LTS](https://askubuntu.com/questions/1404722/downgrade-kernel-for-ubuntu-22-04-lts)
- [参考](https://blog.pmail.idv.tw/?p=19792)

---

感谢阅读！希望这篇文章对你有所帮助。

"""

SD_MOCK_DATA = ""
