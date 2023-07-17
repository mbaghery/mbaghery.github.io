The main question is how to hide those ugly try-catch blocks.
I just hate the look of them.

Say you are trying to save your logs into the cloud.
You can implement this by optimistically call the save method inside a try block, and catching the potential exception:
```
try:
    cloud_instance.save(my_logs)
catch CloudInstanceNotFound as e:
    ...
```
I don't like this for a variety of reasons.
Mainly I think it's very confusing when someone else is reading it: "Why the catch block? What does that exception mean? When is it thrown?"
You'd have to be familiar with the cloud library to know the answers to these questions.
Not everyone is, and not every should have to be.

Secondly, should the connection be dead, if I wanted to establish a connection first before trying to save the logs, I couldn't do that without some level of duplication,
```
try:
    cloud_instance.save(my_logs)
catch CloudInstanceNotFound as e:
    cloud_instance.connect()
    cloud_instance.save(my_logs)
```
or maybe something better like this:
```
try:
    cloud_instance.connect()
catch CloudAlreadyConnected as e:
    # do nothing

cloud_instance.save(my_logs)
```
which still looks extremely ugly in my opinion.
That dangling empty catch block doesn't look too beautiful to me.

Thirdly, what if the same exception was thrown under two different circumstances?
You'd need a whole if-else clause in your catch block. Ugh...

Forthly, sometimes you'd need to revert certain actions in the case of an exception.
Say you were installing 5 packages and the 4th one threw an exception.
You'd probably want to uninstall the first three packages before exiting.
Using the try-catch block, there's no clean way of doing it,
```
try:
    package_installer.install()
catch SomeException:
    package_installer.uninstall()
```
This is ugly, but also, why should we as the user of the `package_installer` object have to remember to revert everything?
Plus, if we had used the `install` method a few more times in other places in our code, we would have to have just as many try-catch blocks doing the same thing.
Also, isn't it really the responsibility of the writer of the `package_installer` class to take care of error handling?

I can't emphasise enough, I just find the whole try-catch thing very distracting.
When I read code, I wanna focus on the main logic, not error handling.



I have come up with four different types of exceptions and ways to deal with them without using explicit try-catch blocks.
Let me know what you think.


Case 1:
If there is a quick way of prechecking if an exception will happen in a certain case, that precheck should be provided by the writer of the package and used by the user.
Example: I'm trying to access the i-th element of an array.
Instead of optimistically grabbing that element and risking getting an exception, I can do:
```
if i < array.length:
    return array[i]
```
While this was a trivial example, it isn't always.

Another example:
```
if item.is_in_stock:
    item.add_to(basket)
```
The alternative would've been to add it without checking and catching an OutOfStock exception... Ugh. No.


Case 2:
Sometimes doing the precheck requires us to do the actual work.
In this case, a caching mechanism should be implemented.
Example: some regex stuff:
```
class EmailAdressParser:
    pattern: str = 'some regex email adress pattern'
    cached_text: str = ''
    cached_match: str = ''

    def parse_email_adress(text):
        if cached_text == text:
            return cached_match
        else:
            result = find(pattern, text)
            if result != None:
                return result
            else:
                throw RegExNotFoundException

    def is_email_adress(text):
        temp_cached_match = find(pattern, text)

        if temp_cached_match != None:
            cached_text = text
            cached_match = temp_cached_match
            return True
        else:
            return False
```
where `find` is a built-in regex function that returns a match if found, and None otherwise.
Now we can easily write:
```
parser = EmailAdressParser()

input_email_address = input from user
if parser.is_email_adress(input_email_address):
    send_email(to=parser.parse_email_adress(input_email_address))
```


Case 3:
Resources such as databases should revert applied changes if an exception occurs.
Database connectivity libraries provide this feature.
In short, when using these libraries, if a statement is executed but not committed, it'll be reverted if an exception happens.
Some languages have a built-in tool for dealing with these things.
Python has a `with` statement that hides the ugly exception handling stuff very nicely. (Not sure if Javascript has a similar thing :| )
A typical statement would look like this:
```
with database.open() as conn:
    conn.execute('insert blah into tblBlah')
```
If the interpreter finishes running everything inside the with statement without any exceptions, we're good.
Otherwise a revert method will be called to revert everything.
The main point is python does that in the background.


Case 4:
Say you want to do a series of tasks, _all_ to be reverted in reverse order in the case of an exception.
In a language like C++ where there's no garbage collection, the user has to deallocate everything manually by defining a delete method.
This method is called by the runtime once the object goes out of scope.
This is an excellent opportunity for undoing the tasks when something goes wrong.
But in Python we'd use the with statement again,
```
with TaskManager() as tm:
    tm.run([task1, task2])
    tm.commit()
```
where `TaskManager` would look something like:
```
class TaskManager:
    def run(tasks):
        for task in tasks:
            task.run()

    def commit():
        for task in tasks:
            task.commit()

    def __enter__():
        ...
        return self

    def __exit__():
        for task in reversed(tasks):
            if task.is_committed:
                task.rollback()
```
where each task implements `run`, `commit`, and `rollback`.
Note that `commit` should never fail.
I repeat, `commit` should never fail.
But that's not a big deal as it literally only changes the value of a boolean variable from false to true.

