## 安装所需包
pip install python-igraph
pip install Theano
## 获取角色相似性存于role_feature之下
python get_role.py
## 获取节点表征存于embed之下
python rolemf.py
## 分类验证
python eval.py