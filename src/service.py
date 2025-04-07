##########################################
# Uvicorn の実行方法（直接実行する場合）
##########################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.app:app", host="0.0.0.0", port=8000, reload=True)