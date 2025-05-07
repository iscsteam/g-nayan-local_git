from config import connection
def data_from_db(query):
    conn = connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    else:
        print("Connection to the database failed.")
        return None


