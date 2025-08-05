# 🛡️ Lucia 的 Git 最小生存手册（本地上传用）

## ✅ 每次改完代码，上传 GitHub：

```bash
# 1. 查看改动状态（可选）
git status

# 2. 添加所有改动（新文件 / 修改 / 删除）
git add .

# 3. 提交（写一句改了啥）
git commit -m "更新了高斯插球逻辑"

# 4. 推送到 GitHub
git push
```

🧠 记不住？背口诀：
> “看一下，全部加，打一锤，送上去。”

---

## 🔁 如果推送时报错，比如 GitHub 连不上：

```bash
git remote -v
```

确认是不是 remote 设置错了  
👉 如果错了：

```bash
git remote remove origin
git remote add origin https://github.com/你的用户名/仓库名.git
```

---

## 🧨 如果你想撤销刚刚的 commit（还没 push）：

```bash
git reset --soft HEAD~1
```

✅ 保留代码内容，撤销提交（改message用）

---

## 💣 如果你想彻底撤销刚刚的 commit + 改动（危险）：

```bash
git reset --hard HEAD~1
```

⚠️ 会丢掉刚刚写的所有代码（小心用）

---

## 🧭 查看历史版本：

```bash
git log --oneline
```

输出示例：

```
54c035f fix insertion bug
3dc0b75 add training pipeline
9e1c2c6 init
```

想查看历史版本内容：

```bash
git checkout 3dc0b75
```

（这是只读状态，不建议长期使用）

---

## 💡 可选：设置 Git 别名（让命令更短）

```bash
git config --global alias.s "status"
git config --global alias.c "commit -m"
git config --global alias.p "push"
git config --global alias.l "log --oneline"
```

以后你可以用：

```bash
git add .
git c "更新了结构"
git p
```

---

