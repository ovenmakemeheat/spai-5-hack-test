mkdir -p ~/.kaggle

cat > ~/.kaggle/kaggle.json << 'EOF'
{
  "username": "microhum",
  "key": "KGAT_be2a1aebf88caf149682374913b3fb70"
}
EOF

chmod 600 ~/.kaggle/kaggle.json

pip install -U kaggle

kaggle datasets list -s dogs