// Copyright (c) 2018 Evan S Weinberg
// Various useful utilities that are common
// among the various test files. 

// Utility one: load gauge configurations if
// they exist in the 'cfgs' directory, otherwise
// use heatbath to generate.

void test_get_gauge_field(complex<double>* gauge_field, Lattice2D* lat_fine, double beta, bool do_free, std::mt19937& generator, bool be_verbose = false, double add_topology = 0.0) {

  // Prepare the gauge field.
  Lattice2D* lat_gauge = new Lattice2D(lat_fine->get_dim_mu(0), lat_fine->get_dim_mu(1), 1); // hack for U(1) fields

  int x_len = lat_fine->get_dim_mu(0);
  //int y_len = lat_fine->get_dim_mu(1);

  if (do_free)
  {
    unit_gauge_u1(gauge_field, lat_gauge);
  }
  else
  {
    bool need_heatbath = false;
    if (beta == 6.0)
    {
      switch (x_len)
      {
        case 32:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l32t32b60_heatbath.dat");
          break;
        case 64:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l64t64b60_heatbath.dat");
          break;
        case 128:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l128t128b60_heatbath.dat");
          break;
        case 192:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l192t192b60_heatbath.dat");
          break;
        case 256:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l256t256b60_heatbath.dat");
          break;
        default:
          need_heatbath = true;
          break;
      }
    }
    else if (beta == 10.0)
    {
      switch (x_len)
      {
        case 32:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l32t32b100_heatbath.dat");
          break;
        case 64:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l64t64b100_heatbath.dat");
          break;
        case 128:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l128t128b100_heatbath.dat");
          break;
        case 192:
          read_gauge_u1(gauge_field, lat_gauge, "./cfgs/l192t192b100_heatbath.dat");
          break;
        default:
          need_heatbath = true;
          break;
      }
    }
    else
      need_heatbath = true;

    if (need_heatbath)
    {
      if (be_verbose) {
        std::cout << "[QMG-NOTE]: L = " << x_len << " beta = " << beta << " requires heatbath generation.\n";
      }

      int n_therm = 4000; // how many heatbath steps to perform.
      int n_meas = 100; // how often to measure the plaquette, topo.
      double* phases = allocate_vector<double>(lat_gauge->get_size_gauge());
      double* tmp_phases = allocate_vector<double>(lat_gauge->get_size_gauge());
      std::uniform_real_distribution<> for_acc_rej(0.0, 1.0);
      random_uniform(phases, lat_gauge->get_size_gauge(), generator, -3.1415926535, 3.1415926535);
      //zero_vector(phases, lat_gauge->get_size_gauge());
      double plaq = 0.0; // track along the way
      double action = 0.0;
      double topo = 0.0;
      for (int i = 0; i < n_therm; i += n_meas)
      {
        int steps = n_therm/n_meas;

        for (int j = 0; j < steps; j += 2) {
          // Perform non-compact updates
          heatbath_noncompact_update(phases, lat_gauge, beta, 2, generator);

          /*double instanton_type = for_acc_rej(generator) < 0.5 ? 1.0 : -1.0;

          for (int k = 0; k < 100; k++) {
            // test creating an instanton
            copy_vector(tmp_phases, phases, lat_gauge->get_size_gauge());
            create_noncompact_instanton_u1(tmp_phases, lat_gauge, instanton_type/10.);

            // set up accept/reject
            double old_action = get_noncompact_action_u1(phases, beta, lat_gauge);
            double new_action = get_noncompact_action_u1(tmp_phases, beta, lat_gauge);

            //printf("%f %f\n", old_action, new_action);

            if (new_action < old_action || for_acc_rej(generator) < exp(old_action - new_action)) {
              copy_vector(phases, tmp_phases, lat_gauge->get_size_gauge());
            }
          }*/

          // Perform 2 non-compact updates
          //heatbath_noncompact_update(phases, lat_gauge, beta, 2, generator);
        }

        //create_noncompact_instanton_u1(phases, lat_gauge, for_acc_rej(generator) < 0.5 ? 1.0 : -1.0);

        // Get compact links.
        polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

        action = get_noncompact_action_u1(phases, beta, lat_gauge);
        plaq = std::real(get_plaquette_u1(gauge_field, lat_gauge));
        topo = std::real(get_topo_u1(gauge_field, lat_gauge));
        if (be_verbose) {
          std::cout << "[QMG-HEATBATH]: Update " << i << " Action " << action << " Plaq " << plaq << " Topo " << topo << "\n";
        }
      }

      // Acquire final gauge field.
      polar_vector(phases, gauge_field, lat_gauge->get_size_gauge());

      // Clean up.
      deallocate_vector(&phases);
      deallocate_vector(&tmp_phases);
    }
  }

  // create an instanton
  if (add_topology != 0)
    create_instanton_u1(gauge_field, lat_gauge, add_topology, x_len/2, x_len/2);

  //double action = get_noncompact_action_u1(phases, lat_gauge);
  double plaq = std::real(get_plaquette_u1(gauge_field, lat_gauge));
  double topo = std::real(get_topo_u1(gauge_field, lat_gauge));
  if (be_verbose) {
    std::cout << "[QMG-GAUGE]: Gauge Plaq " << plaq << " Topo " << topo << "\n";
  }
  delete lat_gauge;
}
