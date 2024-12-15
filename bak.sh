rm -rf /workspace/code/github/auto_openai/*
cp -rf ./* /workspace/code/github/auto_openai
cp -rf ./.gititignore /workspace/code/github/auto_openai
cd /workspace/code/github/auto_openai
git add .
git commit -m "update"
git push origin main
