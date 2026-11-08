# 私密 GitHub 仓库上传说明

## 上传什么

上传解压后 `ssqr-rene-artifact` **目录内的全部内容**，以其中的 `README.md` 作为仓库首页。不要把外层 ZIP、验证用虚拟环境或整个研究工作区一起传入。

根目录保留 `.gitignore`、`.gitattributes`、校验清单、安装说明和复算入口。实验源码、`results/`、`artifact/`、`human_review/`、`docs/`、`provenance/` 均应上传。论文另行提交，不在此仓库中。

## 建立仓库

1. 在 GitHub 建立新仓库，选择 **Private**。不自动选择 MIT 等公开许可证；本包已有保密审阅用途说明及上游许可。
2. 推荐使用 GitHub Desktop：将此目录作为本地仓库，提交全部文件，再发布到刚建立的私密仓库。GitHub Desktop 可保留完整目录结构，并避免浏览器上传的单文件限制。
3. 上传后检查网页能打开 README、人工评审数据和实验表，查看最终文件数与 `provenance/INVENTORY.json` 是否相符。大文件检查结果见发布验收；小型 Git 目录不含大权重；三个自训附件放在同一私库 Release 中，或作为同名审稿附件提供，不作为普通 Git 文件推送。
4. 按投稿系统或会议允许的保密方式提供审稿访问。**仅给私库链接不等于审稿人有访问权限**；需安排邀请或其他获准访问方式。
5. 记录投稿所对应的提交版本；修改后另作新提交。私库和审稿保密用于访问管理，无法从技术上阻止获准访问者复制文件。

## 上线前本地核对

```powershell
python tools/verify_bundle.py
python tools/reproduce_all.py
```

无需将 `reproduced/` 或 `.venv/` 提交；验收记录已放在 `provenance/validation/`。

## 与 RENE 的关系

官网要求清楚说明所依赖的材料并在投稿中提供链接，没有单独的 GitHub 目录模板。IEEE 双栏格式针对论文。当前目录按安装、输入、方法、运行、预期结果、来源和许可组织；发布状态以 provenance/RELEASE_STATUS.json 为准；未投稿或取得 artifact 认证。仓库说明不代替会议论文要求。
