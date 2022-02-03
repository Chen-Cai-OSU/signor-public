import sys
import traceback
import logging
# logger = logging.Logger('catch_all')

def catch_all_logger():
    return logging.Logger('catch_all')

def error_msg():
    ex_type, ex_value, ex_traceback = sys.exc_info()
    trace_back = traceback.extract_tb(ex_traceback)

    # Format stacktrace
    stack_trace = list()

    for i, trace in enumerate(trace_back):
        space = ''*(i+1)
        # msg =  "File : %s \n Line : %d \n Func.Name : %s\n Message : %s\n" % (trace[0], trace[1], trace[2], trace[3])
        msg = f' {space}File({i}): {trace[0]}\n  ' \
            f'{space}Line: {trace[1]}\n  ' \
            f'{space}Func.Name: {trace[2]}\n  ' \
            f'{space}Message: {trace[3]}\n'
        stack_trace.append(msg)

    print()
    for line in stack_trace:
        print(line)
    print("Exception type : %s " % ex_type.__name__)
    print("Exception message : %s" % ex_value)


if __name__ == '__main__':
    print('abc')
    print(5)
    try:
        print(1/0)
    except Exception as e:
        # pprint(stack_trace)
        error_msg()
        catch_all_logger().exception(e)

