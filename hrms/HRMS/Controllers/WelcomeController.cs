using HRMS.Services.Interfaces;
using System.Net;
using Microsoft.AspNetCore.Mvc;

namespace HRMS.Controllers
{
    [Route("")]
    [ApiController]
    public class WelcomeController : ControllerBase
    {
        [HttpGet]
        public string HomePage()
        {
            return "Welcome To HR Management System";
        }
    }
}