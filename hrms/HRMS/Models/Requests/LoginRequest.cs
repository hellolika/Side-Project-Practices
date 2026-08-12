using System.Text.RegularExpressions;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class LoginRequest
    {
        [JsonProperty("Email")]
        public string Email { get; set; }

        [JsonProperty("Password")]
        public string Password { get; set; }
        
        private readonly Regex _emailRegex = new(@"^([\w\.\-]+)@([\w\-]+)((\.(\w){2,3})+)$");

        public void CheckModels()
        {
            if (string.IsNullOrWhiteSpace(Email))
            {
                throw new ApiException(Enum.ApiErrorEnum.InvalidModelState, "Email is required");
            }
            if (!_emailRegex.IsMatch(Email))
            {
                throw new ApiException(Enum.ApiErrorEnum.InvalidModelState, "Invalid email format");
            }
            if (string.IsNullOrWhiteSpace(Password))
            {
                throw new ApiException(Enum.ApiErrorEnum.InvalidModelState, "Password is required");
            }
        }
    }
}