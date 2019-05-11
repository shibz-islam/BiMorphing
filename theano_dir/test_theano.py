import theano
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time


def testCPU():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
            r = f()
            t1 = time.time()
            print("Looping %d times took %f seconds" % (i, t1 - t0))
            print("Result is %s" % (r,))
            if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
                print('Used the cpu')
            else:
                print('Used the gpu')

# basic algebra
def testAlg():
    import numpy
    import theano.tensor as T
    from theano import function
    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    f = function([x, y], z)

    print 'scalar addition'
    print f(2, 3)
    #print(pp(z))
    print(z.eval({x: 16.3, y: 12.1}))

    #adding matricies
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    f = function([x, y], z)

    print '\narray addition'
    print f([[1,2],
      [3,4]],
      [[10,20],
       [30,40]])

    #using numpy array
    x = numpy.array([[1,2],[3,4]])
    y = numpy.array([[10,20],[30,40]])
    print '\nnumpy arrays can be used directly into function as well'
    print f(x,y)

    a = T.vector()
    out = a + a ** 10
    f = function([a], out)
    print '\nvector'
    print f([0, 1, 2])

    b = T.vector()

    out = a ** 2 + b ** 2 + 3 * a * b
    f = function([a, b], out)
    print '\nvectors'
    #f_out = f([1,2],[4,5])
    print f([1,2],[4,5])


def log_func():
    # logistic function (sigmoid)
    x = T.dmatrix('x')
    s = 1 / (1+T.exp(-x))
    logistic = function([x], s)
    x = ([[0,1],[-1,-2]])
    print '\nlogistic function, sigmoid, elemnt wise calc on the matrix'
    print logistic(x)

def multiple_input_output():
    # multiple input, multiple output
    a, b = T.dmatrices('a', 'b') # plural
    diff = a - b
    abs_diff =abs(diff)
    diff_squared = diff**2
    f = function([a, b], [diff, abs_diff, diff_squared])
    print '\nmultiple input, multiple outpur'
    print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

def shared_var():
    # shared var
    from theano import shared
    state = shared(0) # share variable state
    inc = T.iscalar('inc') # int variable inc
    accumulator = function([inc], state, updates=[(state, state+inc)]) # updates parameter takes (shared-variable, new expression)
    print '\nshared var'
    print state.get_value()
    accumulator(1)
    print state.get_value()
    accumulator(300)
    print state.get_value()

    # reset
    state.set_value(-1)
    accumulator(3)
    print state.get_value()

    # another function to call the same share var
    decrementor = function([inc], state, updates=[(state, state-inc)])
    decrementor(2)
    print state.get_value()

def copy_functions():
    # similar functions with diff shared vars
    '''
    state = theano.shared(0)
    inc = T.iscalar('inc')
    >> > accumulator = theano.function([inc], state, updates=[(state, state + inc)], on_unused_input='ignore')
    '''

def gradJacHes():
    # derivative
    x = T.dscalar('x') # scalar
    y = x ** 2
    gy = T.grad(y, x) # T.grad(scalr, can be a list), see below (vector, matrix, ...)
    f = function([x], gy)
    print '\nderivative'
    print f(3.2)

    # gradient (partial derivative w.r.t each element in x vector)
    x = T.vector('x')
    s = T.sum(1 / (1 + T.exp(-x)))
    gs = T.grad(s, x)
    f = function([x], gs)
    print '\ngradient w.r.t vector'
    print f([0, 1])


    # gradient (partial derivative w.r.t each element in x Matrix)
    x = T.dmatrix('x')
    s = T.sum(1 / (1 + T.exp(-x)))
    gs = T.grad(s, x)
    f = function([x], gs)
    print '\ngradient w.r.t matrix'
    print f([[0, 1], [-1, -2]])

    # jacobian (y is a vector, x is a vector) and calc the 1st partial derivative, can use theano.gradient.jacobian() or as below
    # jacobian (not using theano.gradient.jacobian())
    x = T.dvector('x')
    y = x ** 2 # jacobian is
    #print '\nlambda'
    #print lambda i, y, x: T.grad(y[i], [4,4])(i=0)
    J, updates = theano.scan(lambda i, y,x : T.grad(y[i], x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])
    f = theano.function([x], J, updates=updates)
    print '\njacobian'
    print f([4, 4])

    # hessian (y is a scalar, x is a vector) and calc the 2nd partial derivative
    x = T.dvector('x')
    y = x ** 2
    cost = y.sum()
    gy = T.grad(cost, x)
    H, updates = theano.scan(lambda i, gy,x : T.grad(gy[i], x), sequences=T.arange(gy.shape[0]), non_sequences=[gy, x])
    f = theano.function([x], H, updates=updates)
    print '\nhessian'
    print f([4, 4])

def logreg():
    import theano
    import numpy
    import theano.tensor as T

    print "\nlog regression"
    rng = numpy.random

    N = 400
    dim = 784
    D = (rng.randn(N, dim), rng.randint(size=N, low=0, high=2)) # random Data (x, y)
    training_steps = 10000

    # Declare theano symbolic variables
    x = T.matrix("x") # 2-d array, compare with T.tensor3 (3-d) and T.tensor4 (4-d)
    y = T.vector("y")
    w = theano.shared(rng.randn(dim), name="w")
    b = theano.shared(0., name="b")
    print "Initial model (w, b)"
    print w.get_value()
    print b.get_value()

    # Construct theano expression graph
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))         # prob(target=1)
    prediction = p_1 > 0.5                          # prediction threshold
    xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)   # cross-entropy loss function, vector
    cost = xent.mean() + 0.01 * (w ** 2).sum()      # the cost to minimize,  l2 norm ||w||_2
    gw, gb = T.grad(cost, [w, b])

    #Compile
    train_f = theano.function(
                inputs=[x,y],
                outputs=[prediction, xent], # two o/p functions
                updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb))
            )

    # Train
    for i in range(training_steps):
        print D[0]
        pred, err = train_f(D[0], D[1])

    predict_f = theano.function(inputs=[x], outputs=prediction)

    print "Final model:"
    print w.get_value()
    print b.get_value()
    print "target values for D:"
    print D[1]
    print "prediction on D:"
    print predict_f(D[0])


def scanAsForLoop():
    import theano
    import theano.tensor as T
    import numpy as np

    # defining the tensor variables
    X = T.matrix("X")
    W = T.matrix("W")
    b_sym = T.vector("b_sym")

    results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)
    compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=results)

    # test values
    x = np.eye(2, dtype=theano.config.floatX)
    w = np.ones((2, 2), dtype=theano.config.floatX)
    b = np.ones((2), dtype=theano.config.floatX)
    b[1] = 2

    print(compute_elementwise(x, w, b))

    # comparison with numpy
    print(np.tanh(x.dot(w) + b))


if __name__ == '__main__':
    #testCPU()
    #testAlg()
    #log_func()
    #multiple_input_output()
    #shared_var()
    #copy_functions()
    #gradJacHes()
    #scanAsForLoop()
    logreg()



























