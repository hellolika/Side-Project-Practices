using System;
using System.Text.RegularExpressions;
using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
	public class UpdateProfileRequest
	{
        [JsonProperty("TeamId")]
        public int TeamId { get; set; }

        [JsonProperty("WorkLocationId")]
        public int WorkLocationId { get; set; }

        [JsonProperty("Email")]
        public string Email { get; set; }

        [JsonProperty("PhoneNumber")]
        public string PhoneNumber { get; set; }

        [JsonProperty("Address")]
        public string Address { get; set; }

        [JsonProperty("BankAccount")]
        public string BankAccount { get; set; }

        [JsonProperty("Remark")]
        public string Remark { get; set; }
        
        [JsonProperty("BankName")]
        public string BankName { get; set; }
        
        [JsonProperty("Position")]
        public string Position { get; set; }

        private readonly Regex _emailRegex = new(@"^([\w\.\-]+)@([\w\-]+)((\.(\w){2,3})+)$");

        public void CheckModel()
        {
            if (!_emailRegex.IsMatch(Email))
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Invalid email format");
            }
        }

    }
}

