# install requirements
pip3 install -r requirements.txt -q

# add hal commands to the path
mkdir -p ~/bin
cp hal controller.py ~/bin/

# ensure ~/.hal exists
mkdir -p ~/.hal

# add 
cp enable_ipywidgets ~/.hal/enable_ipywidgets

# use fire to magically generate hal autocompletions
./hal -- --completion bash > ~/.hal/completion
head -n 10 ~/.hal/completion > ~/.hal/completion
echo 'bash ~/.hal/completion' >> ~/.zshrc
echo 'export PATH=$PATH":$HOME/bin"' >> ~/.zshrc

zsh ~/.zshrc
