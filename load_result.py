
from utils import load_result, plot_result

def main():
    result_list, task_type = load_result()
    plot_result(result_list, task_type)

if __name__ == "__main__":
    main()