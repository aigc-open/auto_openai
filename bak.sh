project=/workspace/code/github/auto_openai
rm -rf $project/*
cp -rf ./* $project
cp -f ./.gitignore $project
cd $project
git add .
git commit -m "update"
git push origin main

project=/workspace/code/puhua/auto_openai
rm -rf $project/*
cp -rf ./* $project
cp -f ./.gitignore $project
cd $project
git add .
git commit -m "update"
git push origin main
