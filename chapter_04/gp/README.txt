Symbolic regression example
---------------------------(16-May-2022)

RPN expressions:
    binary ops: + - * / ^ %
    unary ops : neg sin cos tan ceil floor
    stack ops : swap dup drop
    other     : nop halt
    symbol    : x
    number    : float, [0,1), fraction of -40 to 40. 

number      operation
---------------------------------------
 <1         fraction of range
  1         +
  2         -
  3         *
  4         /
  5         %
  6         ^
  7         neg
  8         sin
  9         cos
 10         tan
 11         ceil
 12         floor
 13         swap
 14         dup
 15         drop
 16         x
 17         halt
 18         nop

Start with x on the stack, return TOS, stack need not be empty.
Empty stack is error == bad expression

Fix expression length to N.

Objective function: minimize MSE over dataset ala curves.py.


