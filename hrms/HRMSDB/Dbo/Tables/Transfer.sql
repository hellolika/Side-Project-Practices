CREATE TABLE [dbo].[Transfer]
(
	[Id] INT NOT NULL PRIMARY KEY IDENTITY(1,1),
	[MemberId] INT,
	[TransferTypeId] INT,
	[DateCount] DECIMAL(16,9) NULL,
	[Amount] DECIMAL(10,2)  NULL,
	[Remark] NVARCHAR(1000) NULL,
	[PayDate] DATETIME NULL,
    [PayStartDate] DATETIME NULL,
	[PayEndDate] DATETIME NULL,
	[Status] INT NOT NULL DEFAULT 0,
	[IsGenerated] [bit] NOT NULL DEFAULT 0,
    [CreatedBy] INT NOT NULL,
    [CreatedOn] DATETIME NOT NULL DEFAULT GETDATE(),
    [ModifiedBy] INT NOT NULL, 
    [ModifiedOn] DATETIME NOT NULL DEFAULT GETDATE(),
)
