import sys
from db.connection import connection
from .pipeline import retrieve

def main():
  with connection() as conn:
    query = sys.argv[1]
    print(retrieve(conn, query))

if __name__ == "__main__":
    main()


