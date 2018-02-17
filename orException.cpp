#include "orException.h"

using namespace std;
using namespace cvar;

orException::orException(void)
{
}

orException::orException(string msg)
{
	message = msg;
}

void orException::setMessage(string msg)
{
	message = msg;
}


// orCvException
orCvException::orCvException(void)
{
}

orCvException::orCvException(string msg) : orException(msg)
{
}

void orCvException::setFunctionName(string name)
{
	cvfunction = name;
}

void orCvException::setCvExceptionClass(cv::Exception e)
{
	cv_e = e;
}


// orArgException
orArgException::orArgException(void)
{
}

orArgException::orArgException(string msg) : orException(msg)
{
}

void orArgException::setFunctionName(string name)
{
	function = name;
}


// orStateException
orStateException::orStateException(void)
{
}

orStateException::orStateException(string msg) : orException(msg)
{
}

