import copy

class FnArgs:
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self.__dict__)
    
    def update(self, other):
        mergedArgs = copy.deepcopy(self.__dict__)
        mergedArgs.update(other)
        return FnArgs(**mergedArgs)
    

if __name__=='__main__':
    fn_args = FnArgs(
        a='a',
        b='b',
        c='c'
    )
    fn_args.update({'d': 'd'})
    print(fn_args)