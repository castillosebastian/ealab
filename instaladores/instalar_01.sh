#Este script instala en ambiente/ Google Cloud
#Librerias y utilidades basicas de  Ubuntu 22.04
#Lenguajes   Python
#Entornos productivos   VScode
#conectividad a los Storage Buckets de  Google Cloud

#autor            Castillo Claudio S
#email            castilloclaudiosebastiany@gmail.com
#fecha creacion   2023-07-10
#fecha revision   
#Known BUGS       
#ToDo Optims     

mkdir  -p  ~/install
mkdir  -p  ~/log

#instalo Google Cloud SDK
#Documentacion  https://cloud.google.com/sdk/docs/install#deb
sudo  DEBIAN_FRONTEND=noninteractive  apt-get update
sudo  DEBIAN_FRONTEND=noninteractive  apt-get --yes install apt-transport-https ca-certificates gnupg
echo  "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl  https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo  DEBIAN_FRONTEND=noninteractive  apt-get update && sudo DEBIAN_FRONTEND=noninteractive  apt-get --yes install google-cloud-sdk

#instalo paquetes que van a hacer falta para R, Python y Julia
sudo  DEBIAN_FRONTEND=noninteractive  apt-get update  && sudo dpkg --add-architecture  i386

#Herramientas de desarrollo
sudo  DEBIAN_FRONTEND=noninteractive  apt-get --yes install  software-properties-common  build-essential

#librerias necesarias para R, Python, Julia, JupyterLab
sudo  DEBIAN_FRONTEND=noninteractive  apt-get --yes install  libssl-dev    \
      libcurl4-openssl-dev  libxml2-dev    \
      libgeos-dev  libproj-dev             \
      libgdal-dev  librsvg2-dev            \
      ocl-icd-opencl-dev  libmagick++-dev  \
      libv8-dev  libsodium-dev             \
      libharfbuzz-dev  libfribidi-dev      \
      pandoc texlive  texlive-xetex        \
      texlive-fonts-recommended            \
      texlive-latex-recommended            \
      cmake  gdebi  curl  sshpass  nano    \
      vim  htop  iotop  iputils-ping       \
      cron  tmux  git-core  zip  unzip     \
      sysstat  smbclient cifs-utils  rsync

#------------------------------------------------------------------------------

#instalo Google Cloud Fuse  para poder ver el bucket  Version:  0.41.1 | Released:2022-04-28
#Documentacion https://cloud.google.com/storage/docs/gcs-fuse?hl=en-419
gcsfusever="0.41.1"
gcsfusepack="gcsfuse_"$gcsfusever"_amd64.deb"
cd
curl  -L -O "https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v$gcsfusever/$gcsfusepack"
sudo  DEBIAN_FRONTEND=noninteractive  dpkg --install $gcsfusepack
rm   ~/$gcsfusepack


#Preparo para que puedan haber 9 buckets al mismo tiempo
mkdir  -p  ~/buckets
mkdir  -p  ~/buckets/b1
mkdir  -p  ~/buckets/b2
mkdir  -p  ~/buckets/b3
mkdir  -p  ~/buckets/b4
mkdir  -p  ~/buckets/b5
mkdir  -p  ~/buckets/b6
mkdir  -p  ~/buckets/b7
mkdir  -p  ~/buckets/b8
mkdir  -p  ~/buckets/b9


cat > /home/$USER/install/linkear_buckets.sh <<FILE
#!/bin/bash

/snap/bin/gsutil ls | sed -r 's/gs:\/\///' | sed 's/.$//'       \
|  sed 's/^/\/usr\/bin\/gcsfuse --implicit-dirs --file-mode 777 --dir-mode 777    /'    \
|  sed 's/$/ \/home\/$USER\/buckets\/b/'    \
|  awk '{ print \$0, NR}'   \
|  sed -E 's/.(.)$/\1/'   >  /home/$USER/install/linkear_buckets2.sh

chmod  u+x  /home/$USER/install/linkear_buckets2.sh
/home/$USER/install/linkear_buckets2.sh

FILE

chmod  u+x /home/$USER/install/*.sh
chmod  u+x /home/$USER/*.sh


cat > /home/$USER/install/buckets.service  <<FILE
[Unit]
Description=buckets

[Service]
Type=forking
ExecStart=/home/$USER/install/linkear_buckets.sh
WorkingDirectory=/home/$USER/
User=$USER
Group=$USER

[Install]
WantedBy=default.target
FILE

sudo  cp   /home/$USER/install/buckets.service   /etc/systemd/system/

sudo  systemctl daemon-reload


sudo  systemctl enable /etc/systemd/system/buckets.service
sudo  systemctl start  buckets

# systemctl status buckets

#------------------------------------------------------------------------------
#conexiones  a discos compartidos para artifacts de MLFlow
#CAMBIAR USUARIO-CLAVE

sudo  mkdir  -p  /media

sudo  mkdir  -p  /media/mlflow
sudo  chmod 777  /media/mlflow

sudo  mkdir  -p  /media/expshared
sudo  chmod 777  /media/expshared

cat >  /home/$USER/install/mlflowartifacts_cred.txt   <<FILE
username=utn2022
password=overfitting666
FILE

cat >  /home/$USER/install/expshared_cred.txt   <<FILE
username=utn2022
password=overfitting666
FILE


cat >  /home/$USER/install/shareddirs.sh   <<FILE
#!/bin/bash
sudo  mount -t cifs  //utn2022.mlflow.rebelare.com/mlflowartifacts  /media/mlflow     -o credentials=/home/$USER/install/mlflowartifacts_cred.txt,uid=$(id -u),gid=$(id -g),forceuid,forcegid
sudo  mount -t cifs  //utn2022.mlflow.rebelare.com/expshared        /media/expshared  -o credentials=/home/$USER/install/expshared_cred.txt,uid=$(id -u),gid=$(id -g),forceuid,forcegid
FILE

chmod  u+x  /home/$USER/install/shareddirs.sh


cat > /home/$USER/install/shareddirs.service  <<FILE
[Unit]
Description=shared dirs

[Service]
Type=oneshot
ExecStart=/home/$USER/install/shareddirs.sh
WorkingDirectory=/home/$USER/install

[Install]
WantedBy=default.target
FILE

sudo  cp   /home/$USER/install/shareddirs.service   /etc/systemd/system/


sudo  systemctl enable /etc/systemd/system/shareddirs.service
sudo  systemctl daemon-reload
sudo  systemctl start  shareddirs

#sudo systemctl status  shareddirs
#-----------------------------------------------------------

#------------------------------------------------------------------------------
#Instalo Python SIN  Anaconda, Miniconda, etc-------------------------------
#Documentacion  https://docs.python-guide.org/starting/install3/linux/

export PATH="$PATH:/home/$USER/.local/bin"
echo  "export PATH=/home/\$USER/.local/bin:\$PATH"  >>  ~/.bashrc 
source ~/.bashrc 

sudo  apt-get update
sudo  DEBIAN_FRONTEND=noninteractive  apt-get --yes install   python3  python3-pip  python3-dev  ipython3

# Update pip
python3 -m pip install --upgrade pip

# Install general Python packages
python3 -m pip install --user pandas numpy matplotlib fastparquet \
                              pyarrow tables plotly seaborn xlrd \
                              scipy wheel testresources

# Install Machine Learning Python packages
python3 -m pip install --user polars black torch ipykernel \
                              ipywidgets pathlib dill optuna \
                              deap lightgbm hyperopt \
                              
# Install datatable from its repository
python3 -m pip install --user git+https://github.com/h2oai/datatable

# Install specific libraries
python3 -m pip install --user gdown mlflow \
                              nbconvert[webpdf] nb_pdf_template


#--------------------------------------------------------------------
# Installo VSCode Server

# Check if the systemd --user service is enabled and started
if ! systemctl is-active --quiet user@$(id -u); then
    sudo systemctl enable --now user@$(id -u)
fi

# Install code-server
# Documentation: https://coder.com/docs/code-server/latest/install

# Download and unpack the latest release
cd
wget -q https://github.com/cdr/code-server/releases/latest/download/code-server-linux-amd64.tar.gz
tar -xzf code-server-linux-amd64.tar.gz
rm code-server-linux-amd64.tar.gz

# Move it to a location in PATH
sudo mv code-server-linux-amd64 /usr/local/bin/code-server

# Make sure the binary is executable
sudo chmod +x /usr/local/bin/code-server

# Create a systemd service
cat > ~/.config/systemd/user/code-server.service <<FILE
[Unit]
Description=VS Code Server

[Service]
ExecStart=/usr/local/bin/code-server --bind-addr 0.0.0.0:8080 --auth none
Restart=always

[Install]
WantedBy=default.target
FILE

# Enable and start the service
systemctl --user enable code-server
systemctl --user start code-server


#------------------------------------------------------------------------------
#establezco la configuracion del editor  vim

cat >  /home/$USER/.vimrc  <<FILE
set background=dark
colorscheme desert
FILE

#------------------------------------------------------------------------------
#establezco la configuracion de  tmux
#Documentacion  https://gist.github.com/paulodeleo/5594773

cat > /home/$USER/.tmux.conf  <<FILE
# remap prefix from 'C-b' to 'C-a'
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

# split panes using | and -
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# reload config file (change file location to your the tmux.conf you want to use)
bind r source-file ~/.tmux.conf

# switch panes using Alt-arrow without prefix
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

set-option -g status-position top
bind -n F1 next-window

set  -g base-index 1
setw -g pane-base-index 1

set -g mouse on
FILE


#------------------------------------------------------------------------------
#agrego servicio para hacer un git pull del repositorio
# cambio github repo: 'ealab'
cat > /home/$USER/install/repo.sh  <<FILE
#!/bin/bash
cd /home/$USER/ealab
git pull
cd /home/$USER/
FILE

chmod  u+x /home/$USER/install/repo.sh 


cat > /home/$USER/install/repo.service  <<FILE
[Unit]
Description=repo

[Service]
Type=oneshot
ExecStart=/home/$USER/install/repo.sh
WorkingDirectory=/home/$USER/
User=$USER
Group=$USER

[Install]
WantedBy=default.target
FILE

sudo  cp   /home/$USER/install/repo.service   /etc/systemd/system/


sudo  systemctl enable /etc/systemd/system/repo.service
sudo  systemctl daemon-reload


#------------------------------------------------------------------------------
#ULTIMO  paso  Reacomodo la instalacion

sudo  DEBIAN_FRONTEND=noninteractive   apt-get  --yes  update
sudo  DEBIAN_FRONTEND=noninteractive   apt-get  --yes  dist-upgrade
sudo  DEBIAN_FRONTEND=noninteractive   apt-get  --yes  autoremove

#------------------------------------------------------------------------------
