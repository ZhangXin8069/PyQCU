// Copyright (c) 2017 Evan S Weinberg
// Verbosity header.

#ifndef ESW_VERBOSITY
#define ESW_VERBOSITY

#include <iostream>
#include <string>

enum inversion_verbose_level
{
  VERB_PASS_THROUGH = -1, // For preconditioned verbosity only. This means to take whatever's used in verbosity. 
  VERB_NONE = 0, // Inverter prints nothing.
  VERB_SUMMARY = 1, // Inverter prints info about self, total number of iterations at end.
  VERB_RESTART_DETAIL = 2, // SUMMARY + prints residual at each restart (for restarted inverters). Equivalent to SUMMARY for non-restarted version.
  VERB_DETAIL = 3 // RESTART_DETAIL + Prints relative residual at every step. 
};

struct inversion_verbose_struct
{
  inversion_verbose_level verbosity; // How verbose to be.
  std::string verb_prefix; // A string to prefix all verbosity printouts with.
  inversion_verbose_level precond_verbosity; // How verbose the preconditioner should be. 
  std::string precond_verb_prefix; // Prefix for preconditioner. 

  inversion_verbose_struct()
  { }

  inversion_verbose_struct(inversion_verbose_level level, std::string prefix)
    : verbosity(level), verb_prefix(prefix)
  { }

  inversion_verbose_struct(inversion_verbose_level level, std::string prefix, inversion_verbose_level prec_level, std::string prec_prefix)
    : verbosity(level), verb_prefix(prefix),
      precond_verbosity(prec_level), precond_verb_prefix(prec_prefix)
  { }
};

// Shuffles restarted values into current values. 
inline void shuffle_verbosity_restart(inversion_verbose_struct* verb_new, inversion_verbose_struct* verb)
{
  if (verb != 0)
  {
    verb_new->verbosity = (verb->verbosity == VERB_RESTART_DETAIL || verb->verbosity == VERB_SUMMARY) ? VERB_NONE : verb->verbosity; 
    verb_new->verb_prefix = verb->verb_prefix;
    verb_new->precond_verbosity = verb->precond_verbosity;
    verb_new->precond_verb_prefix = verb->precond_verb_prefix; 
  }
  else
  {
    verb_new->verbosity = VERB_NONE;
    verb_new->verb_prefix = "";
    verb_new->precond_verbosity = VERB_NONE;
    verb_new->precond_verb_prefix = "";
  }
}

// Shuffles values for preconditioned solves.
inline void shuffle_verbosity_precond(inversion_verbose_struct* verb_new, inversion_verbose_struct* verb)
{
  if (verb != 0)
  {
    if (verb->precond_verbosity == VERB_PASS_THROUGH)
    {
      verb_new->verbosity = verb->verbosity;
    }
    else
    {
      verb_new->verbosity = verb->precond_verbosity;
    }
    verb_new->verb_prefix = verb->precond_verb_prefix;
    verb_new->precond_verbosity = VERB_NONE;
    verb_new->precond_verb_prefix = "";
  }
  else
  {
    verb_new->verbosity = VERB_NONE;
    verb_new->verb_prefix = "";
    verb_new->precond_verbosity = VERB_NONE;
    verb_new->precond_verb_prefix = "";
  }
  
}

// Properly prints relative residual.
inline void print_verbosity_resid(inversion_verbose_struct* verb, std::string alg, int iter, int ops_count, double relres)
{
  if (verb != 0)
  {
    if (verb->verbosity == VERB_DETAIL)
    {
      std::cout << verb->verb_prefix << alg << " Iter " << iter << " Ops " << ops_count << " RelRes " << relres << "\n";
    }
  }
  
  return; 
}

// Properly prints summary at end of inversion.
inline void print_verbosity_summary(inversion_verbose_struct* verb, std::string alg, bool success, int iter, int ops_count, double relres)
{
  if (verb != 0)
  {
    if (verb->verbosity == VERB_SUMMARY || verb->verbosity == VERB_RESTART_DETAIL || verb->verbosity == VERB_DETAIL)
    {
      std::cout << verb->verb_prefix << alg << " Success " << (success ? "Y" : "N") << " Iter " << iter << " Ops " << ops_count << " RelRes " << relres << "\n";
    }
  }
  
  return; 
}

// Properly prints multishift/rhs summary at end of inversion.
inline void print_verbosity_summary_multi(inversion_verbose_struct* verb, std::string alg, bool success, int iter, int ops_count, double* relres, int n_res)
{
  if (verb != 0)
  {
    if (verb->verbosity == VERB_SUMMARY || verb->verbosity == VERB_RESTART_DETAIL || verb->verbosity == VERB_DETAIL)
    {
      std::cout << verb->verb_prefix << "CG-M " << " Success " << (success? "Y" : "N") << " Iter " << iter << " Ops " << ops_count << " RelRes ";
      for (int n = 0; n < n_res; n++)
      {
        std::cout << relres[n] << " ";
      }

      std::cout << "\n";
    }
  }
  return; 
}

// Properly prints summary at restart.
inline void print_verbosity_restart(inversion_verbose_struct* verb, std::string alg, int iter, int ops_count, double relres)
{
  if (verb != 0)
  {
    if (verb->verbosity == VERB_RESTART_DETAIL || verb->verbosity == VERB_DETAIL)
    {
      std::cout << verb->verb_prefix << alg << " Iter " << iter << " Ops " << ops_count << " RelRes " << relres << "\n";
    }
  }
}


#endif // VERBOSITY